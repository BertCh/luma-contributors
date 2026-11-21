// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

import {
  validateCellDispatchBindings,
  validateCellDispatchOutputs
} from '../cell-indexing/cell-dispatch-validation';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getBoundsReductionNodes} from './bounds-reduction';
import {IS_FINITE_VALUE_WGSL, SPREAD_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

/** Planar coordinates, or longitude/latitude used as plain coordinates (no metric). */
const SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metrics: Object.freeze(['none', 'native'] as const)
});

// Upstream has no Hilbert curve index (the radix sort of the keys is already `GPUSort`).
const OPERATION = 'GPUHilbertKeys';
/** Packed x/y scalar indices and row IDs must remain representable as uint32. */
const MAXIMUM_HILBERT_ITEM_COUNT = 0x7fffffff;

/** Largest Hilbert curve order (bits per axis): the key is a `uint32` of `2 * order` bits. */
export const GPU_HILBERT_MAXIMUM_ORDER = 16;

/**
 * Default curve order. It is one below {@link GPU_HILBERT_MAXIMUM_ORDER} so that the invalid-item
 * key `4^order` stays distinct from every curve cell; order 16 reuses the last cell's key.
 */
export const GPU_HILBERT_DEFAULT_ORDER = 15;

/**
 * Key written for an invalid item (non-finite coordinates or an empty box); invalid items sort
 * last. It is `4^order` (one past the last curve cell) for orders below 16 and `0xffffffff` at
 * order 16, where the 32 key bits are all curve index and the last cell shares the sentinel.
 */
export function getGPUHilbertInvalidKey(order: number): number {
  return order >= GPU_HILBERT_MAXIMUM_ORDER ? 0xffffffff : 4 ** order;
}

/** Number of float32 elements of a {@link GPUHilbertKeys} bounds view. */
export const GPU_HILBERT_BOUNDS_LENGTH = 4;

/** Caller-owned outputs of {@link GPUHilbertKeys}. */
export type GPUHilbertKeysOutput = {
  /** One Hilbert index per item, in `[0, 4^order)`, or {@link getGPUHilbertInvalidKey}. */
  keys: GraphDataView<'uint32'>;
  /**
   * Optional curve-ordered item rows (a stable ascending sort permutation of `keys`). Requires
   * `sortedKeys` or a transient is used for it. Invalid items come last.
   */
  sortedRows?: GraphDataView<'uint32'>;
  /** Optional ascending keys matching `sortedRows`. */
  sortedKeys?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUHilbertKeys}.
 *
 * Per-frame (no recompile): the contents of `positions` or `minima`/`maxima` and `bounds`.
 * Compile-time: `order`, the item count and which outputs are present.
 */
type GPUHilbertKeysSharedProps = {
  /** Prefix for generated node IDs. Defaults to `'hilbert-keys'`. */
  id?: string;
  /**
   * Spatial context of the coordinates; omitted means native planar coordinates. Planar and
   * longitude/latitude (metric `'none'`) are accepted. Longitude/latitude is treated as plain
   * coordinates (no metric, no antimeridian wrap, boxes must have `minX <= maxX`).
   */
  spatialContext?: GPUSpatialContext;
  /** Curve order (bits per axis, 1 to {@link GPU_HILBERT_MAXIMUM_ORDER}). Defaults to {@link GPU_HILBERT_DEFAULT_ORDER}. */
  order?: number;
  /** Point positions to key. Provide this or `minima` and `maxima`. */
  positions?: GraphDataView<'float32x2'>;
  /** Feature box minima; keys are of the box centers. Requires `maxima`. */
  minima?: GraphDataView<'float32x2'>;
  /** Feature box maxima with the same length as `minima`. */
  maxima?: GraphDataView<'float32x2'>;
  /**
   * Per-frame `float32` view `[minX, minY, maxX, maxY]` mapped onto the curve. When omitted, the
   * bounds of the valid items are reduced on the GPU each encoding. Items outside the bounds are
   * clamped to its border cells.
   */
  bounds?: GraphDataView<'float32'>;
  /** Output keys and optional curve-order permutation. */
  output: GPUHilbertKeysOutput;
};

/** Exactly one complete geometry source: positions, or paired box minima/maxima. */
export type GPUHilbertKeysProps = Omit<
  GPUHilbertKeysSharedProps,
  'positions' | 'minima' | 'maxima'
> &
  (
    | {positions: GraphDataView<'float32x2'>; minima?: never; maxima?: never}
    | {positions?: never; minima: GraphDataView<'float32x2'>; maxima: GraphDataView<'float32x2'>}
  );

/**
 * Computes 2D Hilbert curve indices for points or feature-box centers within bounds ("Hilbert
 * keys next to `GPUBVH`") and, optionally, the curve-ordered permutation through the gpgpu
 * radix sort.
 *
 * Definition. The bounds are divided into `2^order x 2^order` cells; an item falls into cell
 * `(floor(u * 2^order), floor(v * 2^order))` clamped to the grid, where `(u, v)` are its
 * normalized coordinates, and its key is the position of that cell on the Hilbert curve (the
 * classic `xy2d` rotation algorithm: the curve starts in the lower-left cell and ends in the
 * lower-right cell; it first moves along +y for odd orders and along +x for even orders). Consecutive keys are edge-adjacent cells, which makes the order
 * better at keeping neighbours together than the Morton (Z) order.
 *
 * Keys are integer exact (u32 arithmetic) once the cell is chosen. The cell comes from f32
 * arithmetic `(value - minimum) / extent`, so values within an f32 rounding of a cell border may
 * land in either neighbouring cell. Degenerate bounds (zero extent) put every item in cell 0 on
 * that axis. Invalid items (non-finite coordinates, or boxes with `minimum > maximum`) get
 * {@link getGPUHilbertInvalidKey}; at order 16 that value is also the key of the last curve cell,
 * so an invalid item can tie with it (ties keep ascending row order).
 *
 * Cost: one pass over the items (plus one single-workgroup bounds reduction without `bounds`) and,
 * with `sortedRows`, a radix sort over `2 * order + 1` key bits. No atomics; deterministic.
 *
 * Coordinates only, no metric: `spatialContext` accepts planar coordinates or longitude/latitude with
 * metric `'none'`; great-circle and other metric contexts throw. Longitude/latitude degrees are used
 * as plain coordinates, so nothing wraps at the antimeridian (a box must have `minX <= maxX`).
 */
export class GPUHilbertKeys implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUHilbertKeysProps;
  /** Resolved spatial context (planar or longitude/latitude without metric). */
  readonly spatialContext: GPUSpatialContext;
  /** Curve order. */
  readonly order: number;
  /** Number of items. */
  readonly itemCount: number;

  constructor(props: GPUHilbertKeysProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'order',
      'positions',
      'minima',
      'maxima',
      'bounds',
      'output'
    ]);
    const capturedOutput = snapshotConstructionFields(captured.output, [
      'keys',
      'sortedRows',
      'sortedKeys'
    ]);
    props = Object.freeze({
      ...captured,
      ...(captured.spatialContext
        ? {spatialContext: snapshotConstructionSpatialContext(captured.spatialContext)}
        : {}),
      output: Object.freeze({...capturedOutput})
    });
    const id = props.id ?? 'hilbert-keys';
    this.id = id;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.order = props.order ?? GPU_HILBERT_DEFAULT_ORDER;
    validateCapacity(id, 'order', this.order, 1, GPU_HILBERT_MAXIMUM_ORDER);
    const {positions, minima, maxima, bounds} = props;
    if (positions) {
      if (minima || maxima) {
        throw new Error(`${id} takes positions or minima and maxima, not both`);
      }
      validateSinglePackedView(positions, ['float32x2'], `${id} positions`);
      this.itemCount = positions.length;
    } else {
      if (!minima || !maxima) {
        throw new Error(`${id} needs positions, or minima and maxima`);
      }
      validateSinglePackedView(minima, ['float32x2'], `${id} minima`);
      validateSinglePackedView(maxima, ['float32x2'], `${id} maxima`);
      if (minima.length !== maxima.length) {
        throw new Error(`${id} minima and maxima must have the same length`);
      }
      this.itemCount = minima.length;
    }
    validateCapacity(id, 'itemCount', this.itemCount, 1, MAXIMUM_HILBERT_ITEM_COUNT);
    if (bounds) {
      validateSinglePackedView(bounds, ['float32'], `${id} bounds`);
      if (bounds.length < GPU_HILBERT_BOUNDS_LENGTH) {
        throw new Error(`${id} bounds must hold ${GPU_HILBERT_BOUNDS_LENGTH} float32 values`);
      }
    }
    const {keys, sortedRows, sortedKeys} = props.output;
    for (const [name, view] of [
      ['keys', keys],
      ['sortedRows', sortedRows],
      ['sortedKeys', sortedKeys]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
        if (view.length !== this.itemCount) {
          throw new Error(`${id} output.${name} must have one row per item`);
        }
      }
    }
    if (sortedKeys && !sortedRows) {
      throw new Error(`${id} output.sortedKeys needs output.sortedRows`);
    }
    validateCellDispatchOutputs(id, {keys, sortedRows, sortedKeys}, []);
    validateGraphOutputsDisjointFromInputs(
      id,
      [keys, sortedRows, sortedKeys],
      [positions, minima, maxima, bounds]
    );
  }

  /** Returns the optional bounds reduction, the key kernel and the optional sort nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, order, itemCount} = this;
    const {positions, minima, maxima} = props;
    const {keys, sortedRows, sortedKeys} = props.output;
    const partialCount = Math.min(256, Math.max(1, Math.ceil(itemCount / 4096)));
    validateCellDispatchBindings(
      graph,
      id,
      {
        positions,
        minima: positions ? undefined : minima,
        maxima: positions ? undefined : maxima,
        bounds: props.bounds,
        keys,
        sortedRows,
        sortedKeys
      },
      [
        ...(!props.bounds
          ? ([
              ['bounds scratch', 16],
              ['bounds partial scratch', partialCount > 1 ? partialCount * 16 : 0]
            ] as const)
          : []),
        ...(sortedRows
          ? ([
              ['sort scratch', Math.max(itemCount, 1) * 4],
              ['radix histogram scratch', itemCount > 1 ? 16 * Math.ceil(itemCount / 256) * 4 : 0]
            ] as const)
          : [])
      ]
    );
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      minima,
      maxima,
      props.bounds,
      keys,
      sortedRows,
      sortedKeys
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const itemBindings: WGSLKernelBinding[] = positions
      ? [{name: 'positions', view: positions, type: 'f32', access: 'read'}]
      : [
          {name: 'minima', view: minima!, type: 'f32', access: 'read'},
          {name: 'maxima', view: maxima!, type: 'f32', access: 'read'}
        ];
    const declarations = `${IS_FINITE_VALUE_WGSL}
// Returns (x, y, valid, 0) for an item: the point, or the box center (halved before the sum to
// avoid overflow near FLT_MAX).
fn readItem(row: u32) -> vec4f {
${
  positions
    ? `  let point = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  let valid = isFiniteValue(point.x) && isFiniteValue(point.y);
  return vec4f(point, select(0.0, 1.0, valid), 0.0);`
    : `  let minimum = vec2f(minima[minimaOffset + row * 2u], minima[minimaOffset + row * 2u + 1u]);
  let maximum = vec2f(maxima[maximaOffset + row * 2u], maxima[maximaOffset + row * 2u + 1u]);
  let valid = isFiniteValue(minimum.x) && isFiniteValue(minimum.y) &&
    isFiniteValue(maximum.x) && isFiniteValue(maximum.y) &&
    minimum.x <= maximum.x && minimum.y <= maximum.y;
  return vec4f(minimum * 0.5 + maximum * 0.5, select(0.0, 1.0, valid), 0.0);`
}
}`;

    let bounds = props.bounds;
    if (!bounds) {
      bounds = createTransientView(graph, `${id}-bounds`, 'float32', GPU_HILBERT_BOUNDS_LENGTH);
      nodes.push(
        ...getBoundsReductionNodes<Parameters>(graph, {
          id,
          operation: OPERATION,
          variant: positions ? 'bounds-points' : 'bounds-boxes',
          bindings: itemBindings,
          declarations,
          itemCount,
          output: bounds
        })
      );
    }

    const rows = sortedRows
      ? createTransientView(graph, `${id}-rows`, 'uint32', itemCount)
      : undefined;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: `${positions ? 'points' : 'boxes'}-order-${order}${rows ? '-rows' : ''}`,
        bindings: [
          ...itemBindings,
          {name: 'bounds', view: bounds, type: 'f32', access: 'read'},
          {name: 'keysOut', view: keys, type: 'u32', access: 'read_write'},
          ...(rows
            ? [{name: 'rowsOut', view: rows, type: 'u32' as const, access: 'read_write' as const}]
            : [])
        ],
        invocationCount: itemCount,
        declarations: `${declarations}
${SCALED_QUOTIENT_WGSL}
const ORDER: u32 = ${order}u;
const CELLS: u32 = ${2 ** order}u;

fn quantizeAxis(value: f32, minimum: f32, maximum: f32) -> u32 {
  let scale = max(abs(minimum), abs(maximum));
  if (!(maximum > minimum) || !isFiniteValue(scale)) { return 0u; }
  // Power-of-two normalization prevents both overflowing differences and reciprocal flushing
  // under Metal fast math. The shared quotient divides only bounded significands.
  let exponent = frexp(scale).exp;
  let lower = ldexp(minimum, -exponent);
  let upper = ldexp(maximum, -exponent);
  let position = ldexp(clamp(value, minimum, maximum), -exponent);
  let normalized = clamp(getScaledQuotient(position - lower, upper - lower, 1.0, 0), 0.0, 1.0);
  return min(u32(floor(normalized * f32(CELLS))), CELLS - 1u);
}

// Spreads the low 16 bits of a value to the even bit positions.
${SPREAD_BITS_WGSL}

// The Hilbert curve index of cell (x, y) on a CELLS x CELLS grid, equal to the classic xy2d
// rotation loop. The loop carries a rotate/flip state from the top bit down, so its 16 steps
// are serial and branch on data. The state transform of each bit level is a small affine map over
// GF(2), so the same result comes from a parallel prefix over the levels (Hacker's Delight 16-5,
// as in rawrunprotected's "Hilbert curves in O(log n)"): four branch-free rounds on 16-bit lanes,
// then one interleave.
fn getHilbertIndex(cellX: u32, cellY: u32) -> u32 {
  let x = cellX << (16u - ORDER);
  let y = cellY << (16u - ORDER);
  let a0 = x ^ y;
  let b0 = 0xffffu ^ a0;
  let c0 = 0xffffu ^ (x | y);
  let d0 = x & (y ^ 0xffffu);
  var transformA = a0 | (b0 >> 1u);
  var transformB = (a0 >> 1u) ^ a0;
  var transformC = ((c0 >> 1u) ^ (b0 & (d0 >> 1u))) ^ c0;
  var transformD = ((a0 & (c0 >> 1u)) ^ (d0 >> 1u)) ^ d0;
  for (var shift = 2u; shift <= 4u; shift = shift << 1u) {
    let a = transformA;
    let b = transformB;
    let c = transformC;
    let d = transformD;
    transformA = (a & (a >> shift)) ^ (b & (b >> shift));
    transformB = (a & (b >> shift)) ^ (b & ((a ^ b) >> shift));
    transformC = transformC ^ ((a & (c >> shift)) ^ (b & (d >> shift)));
    transformD = transformD ^ ((b & (c >> shift)) ^ ((a ^ b) & (d >> shift)));
  }
  let a = transformA;
  let b = transformB;
  let c = transformC;
  let d = transformD;
  transformC = c ^ ((a & (c >> 8u)) ^ (b & (d >> 8u)));
  transformD = d ^ ((b & (c >> 8u)) ^ ((a ^ b) & (d >> 8u)));
  let low = transformC ^ (transformC >> 1u);
  let high = transformD ^ (transformD >> 1u);
  let indexLow = x ^ y;
  let indexHigh = high | (0xffffu ^ (indexLow | low));
  return ((spreadBits(indexHigh) << 1u) | spreadBits(indexLow)) >> (32u - 2u * ORDER);
}`,
        body: `${rows ? 'rowsOut[rowsOutOffset + index] = index;' : ''}
  let item = readItem(index);
  var key = ${getGPUHilbertInvalidKey(order)}u;
  if (item.z > 0.5) {
    let cellX = quantizeAxis(item.x, bounds[boundsOffset], bounds[boundsOffset + 2u]);
    let cellY = quantizeAxis(item.y, bounds[boundsOffset + 1u], bounds[boundsOffset + 3u]);
    key = getHilbertIndex(cellX, cellY);
  }
  keysOut[keysOutOffset + index] = key;`
      })
    );

    if (sortedRows && rows) {
      const orderedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', itemCount);
      const orderedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', itemCount);
      nodes.push(
        ...new GPUSort({
          id: `${id}-sort`,
          keys,
          values: rows,
          outputKeys: orderedKeys,
          outputValues: orderedRows,
          algorithm: 'radix',
          keyBits: Math.min(32, 2 * order + 1)
        }).getCommandNodes(graph)
      );
      // Published sort destinations must use separate buffers. Publish each caller column in
      // its own dispatch so disjoint packed regions of one caller buffer remain valid.
      for (const [name, input, output] of [
        ['rows', orderedRows, sortedRows],
        ['keys', orderedKeys, sortedKeys]
      ] as const) {
        if (output)
          nodes.push(
            createWGSLKernelNode(graph, {
              id: `${id}-publish-${name}`,
              operation: OPERATION,
              invocationCount: itemCount,
              bindings: [
                {name: 'input', view: input, type: 'u32', access: 'read'},
                {name: 'output', view: output, type: 'u32', access: 'read_write'}
              ],
              body: 'output[outputOffset + index] = input[inputOffset + index];'
            })
          );
      }
    }
    return nodes;
  }
}
