// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolveCellPrecision, type CellPrecision} from './cell-precision';
import {
  resolveCellLongitudePolicy,
  CELL_WRAP_LONGITUDE_WGSL,
  type CellLongitudePolicy
} from './cell-longitude-policy';
import {snapshotCellConstruction} from './cell-construction-snapshot';
import {
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createCellRowKernelNode,
  type CellRowKernelBinding
} from '../cell-aggregation/cell-row-kernel';
import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getGraphViewChunks,
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphViewTopology,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateCellWritableViews} from '../cell-aggregation/cell-validation';
import {CELL_KEY_WGSL, getCellKeyLayout, QUADBIN_TILE_WGSL} from '../cell-aggregation/cell-keys';
import {validateCellIndexResolution, type GPUCellIndexFamily} from './cell-index-families';
import {GEOHASH_INDEX_WGSL, QUADKEY_INDEX_WGSL, S2_INDEX_WGSL} from './cell-index-wgsl';
import {H3_INDEX_WGSL} from './h3-index-wgsl';
import {H3_INDEX_EXTENDED_WGSL} from './h3-index-extended-wgsl';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUPointToCell';

/** Row indices and dispatch counts are unsigned 32-bit integers. */
const MAXIMUM_POINT_ROW_COUNT = 0xffffffff;

/** Packed or record-batch position input. */
type PointPositionRows = GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;

/** Caller-owned outputs of {@link GPUPointToCell}, packed for packed input and chunked otherwise. */
export type GPUPointToCellOutput<Positions extends PointPositionRows = GraphDataView<'float32x2'>> =
  {
    /** One 64-bit key per row as little-endian `(low, high)` words. Zero for masked or non-finite rows. */
    cells: Positions extends GraphVectorView
      ? GraphVectorView<'uint32x2'>
      : GraphDataView<'uint32x2'>;
    /** Optional `1` for every row that received a key and `0` for masked, non-finite or unplaceable rows. */
    validity?: Positions extends GraphVectorView
      ? GraphVectorView<'uint32'>
      : GraphDataView<'uint32'>;
  };

/**
 * Properties for {@link GPUPointToCell}.
 *
 * Per-frame (no recompile): the contents of `positions` and `mask`. Topology (needs a new graph):
 * `family`, `resolution`, `precision`, `longitudePolicy`, the row count, chunk boundaries and which optional views are present.
 */
export type GPUPointToCellProps<Positions extends PointPositionRows = GraphDataView<'float32x2'>> =
  {
    /** Prefix for generated node IDs. Defaults to `'point-to-cell'`. */
    id?: string;
    /** Grid family of the output keys. */
    family: GPUCellIndexFamily;
    /**
     * Resolution of the output cells, fixed at compile time: Quadbin 0-26, H3 0-15, quadkey 1-29,
     * geohash 1-12, S2 0-30.
     */
    resolution: number;
    /** H3 arithmetic: float32 by default, or double-single (hi/lo float32 pairs).
     * Other families support only float32 indexing. Inputs and outputs remain float32/uint32.
     */
    precision?: CellPrecision;
    /** Longitude policy, default 'family-default'; 'wrap' uses [-180, 180) degrees for every family. */
    longitudePolicy?: CellLongitudePolicy;
    /**
     * Coordinate contract of `positions`. Defaults to, and must be, longitude/latitude without a
     * metric; planar contexts throw (project-first does not apply: grids index geographic positions).
     */
    spatialContext?: GPUSpatialContext;
    /** Longitude/latitude degrees per row. */
    positions: Positions;
    /** Optional per-row mask; zero produces a zero key. */
    mask?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
    /** Caller-owned outputs. */
    output: GPUPointToCellOutput<Positions>;
  };

/**
 * Converts longitude/latitude points to 64-bit discrete global grid cell keys, one row per row.
 *
 * One compute kernel per input chunk writes `output.cells` (little-endian `(low, high)` words,
 * the Arrow `Uint64` layout that `GPUCellAggregation` accepts as pre-keyed `cells`) and optionally `output.validity`.
 * Rows that are masked out or have a non-finite longitude or latitude get the zero key, the "no
 * cell" sentinel of every family. The contributor never compiles, submits or reads back.
 *
 * `longitudePolicy: 'wrap'` first maps longitude to [-180, 180) degrees for every family.
 * By default each family's native clipping/wrapping below applies. The direct contributor
 * requires at least one position row; named graph forms also support zero rows without dispatch.
 *
 * Exactness per family, with positions rounded to f32:
 * - `'quadbin'`, `'quadkey'`: integer-only arithmetic (the `GPUCellAggregation` fixed-point
 *   Mercator row, exact column), bit-identical on every device and equal to the BigInt reference.
 *   Longitude 180 wraps to column 0 and latitude is clipped to ±85.051129. Quadkey tiles are the
 *   Quadbin tiles (same clip semantics), packed as documented on {@link GPUCellIndexFamily}.
 *   Against the f64 Mercator formula, measured row mismatches were 1.9e-4 at zoom 16,
 *   1.3% at zoom 26 and 9.6% at zoom 29 (about 2% at
 *   zoom 27, 4.5% at zoom 28); the kernel itself stays deterministic.
 * - `'geohash'`: integer-exact. The bisection of an f32 coordinate is an exact dyadic floor, so
 *   the key equals the reference bisection of the same f32 value. Longitude 180 and latitude 90
 *   land in the last bin; larger inputs clamp.
 * - `'s2'`: f32 face, `uv`, quadratic `st` and `ij`, then the integer Hilbert state machine. The
 *   result equals an f64 reference except for points within the f32 error of a cell edge (about
 *   6e-8 in `st`), so the mismatch rate grows with the level. Measured on 10k equal-area points:
 *   zero mismatches in that sample through level 10 (an observation, not an exactness guarantee),
 *   1e-4 at level 11, 5e-4 at 12, 2.6e-3 at 15, 6.6e-2 at 20, 0.26 at 22,
 *   0.78 at 24 and nearly every point at levels 26-30. Through level 22 a mismatch is always an
 *   edge neighbor of the reference cell; beyond it the error spans several cells (about 9 cells at
 *   level 26 and 140 at level 30). No level is guaranteed exact at every cell edge; choose a
 *   level and precision according to the required tolerance for boundary misclassification.
 * - `'h3'`: f32 `latLngToCell` (see `H3_INDEX_WGSL`). Exact against h3-js at resolutions 0-4;
 *   the mismatch rate on uniform points is about 1e-3 at resolution 9, 1.8% at resolution 12 and
 *   28% at resolution 15, and a mismatch is always a grid neighbor. With `precision: 'double-single'`
 *   (double-float arithmetic, a few times slower) the key equals h3-js evaluated on the same f32
 *   input positions at every resolution 0-15. Verified with zero mismatches against h3-js on
 *   20000 uniform points, 6000 near-cell-edge points, 1212 near-pentagon points and the pole,
 *   antimeridian and equator probes per resolution (headless WebGPU on Apple Metal, spec
 *   `h3-index-extended.spec.ts`). The f32 input positions are still quantized (about 1e-5 degrees at
 *   longitude 180, roughly a metre), so at fine resolutions the key is the cell of the rounded
 *   position, not of the original double-precision coordinate. The arithmetic relies on the
 *   shader compiler preserving error-free transformations (guarded with an integer OR of a
 *   runtime zero); it has been verified on Metal only, so check other backends before relying on it.
 *
 * Accepts chunked input. Every chunk must be packed; `mask` and row-aligned outputs must
 * mirror the position chunk boundaries, as in `GPUTimeWindowFilter`. Packed input keeps packed
 * outputs. No input repacking or row-proportional scratch is needed. Outputs must not share
 * buffers with the inputs. Output byte ranges must not overlap across chunks or columns.
 */
export class GPUPointToCell<Positions extends PointPositionRows = GraphDataView<'float32x2'>>
  implements GPUCommandNodeProducer
{
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointToCellProps<Positions>;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  /** One parameter upload plus one key dispatch per nonempty input chunk. */
  readonly estimatedCommandNodeCount: number;
  /** Resolved spatial context (longitude/latitude, no metric). */
  readonly spatialContext: GPUSpatialContext;

  private readonly precision: 'float32' | 'double-single';

  constructor(props: GPUPointToCellProps<Positions>) {
    props = snapshotCellConstruction(props, [
      'id',
      'family',
      'resolution',
      'precision',
      'longitudePolicy',
      'spatialContext',
      'positions',
      'mask',
      ['output', ['cells', 'validity']]
    ]);
    this.id = props.id ?? 'point-to-cell';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
      GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT
    );
    const id = this.id;
    validateCellIndexResolution(id, props.family, props.resolution);
    this.precision = resolveCellPrecision(id, props.precision, props.family === 'h3');
    resolveCellLongitudePolicy(id, props.longitudePolicy);
    validateCapacity(id, 'positions rows', props.positions.length, 1, MAXIMUM_POINT_ROW_COUNT);
    for (const [name, view, formats] of [
      ['positions', props.positions, ['float32x2']],
      ['mask', props.mask, ['uint32']],
      ['output.cells', props.output.cells, ['uint32x2']],
      ['output.validity', props.output.validity, ['uint32']]
    ] as const) {
      if (view) {
        for (const chunk of getGraphViewChunks(view as GraphDataView | GraphVectorView)) {
          validateSinglePackedView(chunk, formats, `${id} ${name}`);
        }
        validateRequirement(
          id,
          view.length === props.positions.length,
          `${name} length must equal the row count`
        );
        validateGraphViewTopology(id, name, props.positions, view);
      }
    }
    validateCellWritableViews(id, [
      ['output.cells', props.output.cells],
      ['output.validity', props.output.validity]
    ]);
    this.chunkPlan = planGraphChunkDispatch(id, {
      positions: props.positions,
      mask: props.mask,
      cells: props.output.cells,
      validity: props.output.validity
    });
    this.estimatedCommandNodeCount =
      1 + getGraphViewChunks(props.positions).filter(chunk => chunk.length > 0).length;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.cells, props.output.validity],
      [props.positions, props.mask]
    );
  }

  /** Returns the runtime parameter upload and one key dispatch per nonempty position chunk. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {family, resolution, output} = props;
    const isExtended = this.precision === 'double-single';
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.mask,
      output.cells,
      output.validity
    ]);
    for (const view of [props.positions, props.mask, output.cells, output.validity]) {
      for (const chunk of view ? getGraphViewChunks(view as GraphDataView | GraphVectorView) : []) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          'row chunk',
          getViewBindingRange(chunk).size,
          'reduce the number of rows per chunk'
        );
        validateCapacity(
          id,
          'row chunk buffer byteLength',
          chunk.buffer.byteLength,
          0,
          graph.device.limits.maxBufferSize
        );
      }
    }
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    const parameters = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    const nodes: GPUCommandNode<Parameters>[] = [...parameters.nodes];
    const positionChunks = getGraphViewChunks(props.positions);
    const maskChunks = props.mask ? getGraphViewChunks(props.mask) : undefined;
    const cellChunks = getGraphViewChunks(output.cells);
    const validityChunks = output.validity ? getGraphViewChunks(output.validity) : undefined;
    for (const [chunkIndex, positions] of positionChunks.entries()) {
      if (positions.length === 0) {
        continue;
      }
      const mask = maskChunks?.[chunkIndex];
      const validity = validityChunks?.[chunkIndex];
      const bindings: CellRowKernelBinding[] = [
        {name: 'positions', view: positions, components: 2, type: 'f32', access: 'read'},
        ...(mask
          ? [{name: 'rowMask', view: mask, components: 1, type: 'u32', access: 'read'} as const]
          : []),
        {
          name: 'cellsOut',
          view: cellChunks[chunkIndex],
          components: 2,
          type: 'u32',
          access: 'read_write'
        },
        ...(validity
          ? [
              {
                name: 'validityOut',
                view: validity,
                components: 1,
                type: 'u32',
                access: 'read_write'
              } as const
            ]
          : [])
      ];
      const keyExpression = getKeyExpression(family, resolution, isExtended);
      nodes.push(
        createCellRowKernelNode<Parameters>(graph, {
          id: positionChunks.length === 1 ? `${id}-keys` : `${id}-keys-chunk-${chunkIndex}`,
          operation: OPERATION,
          variant: isExtended ? 'h3-extended' : family,
          bindings,
          length: positions.length,
          parameters: parameters.slots[chunkIndex],
          declarations: `const RESOLUTION: u32 = ${resolution}u;
${getFamilyDeclarations(family, isExtended)}
${IS_FINITE_BITS_WGSL}
${props.longitudePolicy === 'wrap' ? CELL_WRAP_LONGITUDE_WGSL : ''}`,
          body: `var longitude = positions[positionsOffset + 2u * index];
  let latitude = positions[positionsOffset + 2u * index + 1u];
  var isValid = isFiniteBits(longitude) && isFiniteBits(latitude);
  ${mask ? 'isValid = isValid && rowMask[rowMaskOffset + index] != 0u;' : ''}
  var key = vec2u(0u);
  if (isValid) {
    ${props.longitudePolicy === 'wrap' ? 'longitude = cellWrapLongitudeDegrees(longitude);' : ''}
    key = ${keyExpression};
  }
  isValid = isValid && any(key != vec2u(0u));
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  ${validity ? 'validityOut[validityOutOffset + index] = select(0u, 1u, isValid);' : ''}`
        })
      );
    }
    return nodes;
  }
}

function getFamilyDeclarations(family: GPUCellIndexFamily, isExtended: boolean): string {
  switch (family) {
    case 'quadbin':
      return `${CELL_KEY_WGSL}\n${QUADBIN_TILE_WGSL}`;
    case 'quadkey':
      return QUADKEY_INDEX_WGSL;
    case 'geohash':
      return GEOHASH_INDEX_WGSL;
    case 's2':
      return S2_INDEX_WGSL;
    case 'h3':
      return isExtended
        ? `${dggs.source}\n${H3_INDEX_WGSL}\n${H3_INDEX_EXTENDED_WGSL}`
        : `${dggs.source}\n${H3_INDEX_WGSL}`;
  }
}

function getKeyExpression(
  family: GPUCellIndexFamily,
  resolution: number,
  isExtended: boolean
): string {
  switch (family) {
    case 'quadbin': {
      const layout = getCellKeyLayout('quadbin', resolution);
      return `cellGetKey(
      quadbinGetCompactKey(quadbinGetTileX(longitude, RESOLUTION), quadbinGetTileY(latitude, RESOLUTION)),
      ${layout.headerHigh}u, RESOLUTION, ${layout.lowBit}u)`;
    }
    case 'quadkey':
      return 'cellIndexQuadkeyFromLngLat(longitude, latitude, RESOLUTION)';
    case 'geohash':
      return 'cellIndexGeohashFromLngLat(longitude, latitude, RESOLUTION)';
    case 's2':
      return 'cellIndexS2FromLngLat(longitude, latitude, RESOLUTION)';
    case 'h3':
      return `${isExtended ? 'cellIndexH3ExtendedFromLngLat' : 'cellIndexH3FromLngLat'}(vec2f(longitude, latitude), RESOLUTION)`;
  }
}
