// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {HEALPIX_INDEX_WGSL, HEALPIX_MAXIMUM_ORDER} from './healpix-index-wgsl';

const OPERATION = 'GPUHealpixPointToCell';

/**
 * Index scheme of the {@link GPUHealpixPointToCell} keys. `'nested'` is the plain nested pixel
 * number (what healpix-geo `nested` returns), in which cell 0 of face 0 has key 0, the "no cell"
 * sentinel of the other families, so read `output.validity`. `'uniq'` adds `4^(order + 1)` (the
 * IVOA UNIQ scheme): never zero, and the order is part of the key.
 */
export type GPUHealpixIndexScheme = 'nested' | 'uniq';

/** Caller-owned outputs of {@link GPUHealpixPointToCell}. */
export type GPUHealpixPointToCellOutput = {
  /** One 64-bit key per row as little-endian `(low, high)` words (the Arrow `Uint64` layout). Zero for masked or non-finite rows. */
  cells: GraphDataView<'uint32x2'>;
  /** Optional `1` for every row that received a key and `0` for masked or non-finite rows. */
  validity?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUHealpixPointToCell}.
 *
 * Per-frame (no recompile): the contents of `positions` and `mask`. Compile-time: `order`,
 * `scheme`, the row count and which optional views exist.
 */
export type GPUHealpixPointToCellProps = {
  /** Prefix for generated node IDs. Defaults to `'healpix-point-to-cell'`. */
  id?: string;
  /** HEALPix order (depth), 0 to 29: `nside = 2^order`, `12 * 4^order` cells. */
  order: number;
  /** Key scheme. Defaults to `'nested'`. */
  scheme?: GPUHealpixIndexScheme;
  /** Coordinate contract of `positions`: longitude/latitude without a metric (the default). */
  spatialContext?: GPUSpatialContext;
  /** Longitude/latitude degrees per row. */
  positions: GraphDataView<'float32x2'>;
  /** Optional per-row mask; zero produces a zero key. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUHealpixPointToCellOutput;
};

/**
 * Converts longitude/latitude points to HEALPix nested cell keys (RTS-08), one row per row, with
 * the same output layout as `GPUPointToCell` (little-endian `(low, high)` words and an optional
 * validity flag; masked and non-finite rows get the zero key).
 *
 * The projection is the closed-form `ang2pix_nest` of Gorski et al. (2005) on the sphere; the
 * cells have equal area. There is no ellipsoid: healpix-geo's `ellipsoid='sphere'` is the
 * reference. Geographic latitude on a spheroid needs the authalic conversion first, which this
 * contributor does not do.
 *
 * **float32 accuracy.** Positions are rounded to float32 and the projection is evaluated in f32,
 * so a point can fall in a neighbouring cell when it lies within the rounding error of a cell
 * edge. The error is a few units in `2^-23` of `4 * nside` cell widths, so the share of wrong
 * cells grows linearly with `nside`; the spec measures it against healpix-geo on float32 inputs
 * and the documentation lists the rates by order. Orders above about 14 (cells under 400 m) are
 * dominated by the 1.5e-5 degree quantization of float32 longitude near 180 degrees: the key is
 * the cell of the rounded position. A mismatch is always an adjacent cell.
 *
 * Inputs must be single packed views. Outputs must not share buffers with the inputs.
 */
export class GPUHealpixPointToCell implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUHealpixPointToCellProps;
  /** Resolved spatial context (longitude/latitude, no metric). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUHealpixPointToCellProps) {
    this.id = props.id ?? 'healpix-point-to-cell';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext ?? GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
      GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT
    );
    validateCapacity(id, 'order', props.order, 0, HEALPIX_MAXIMUM_ORDER);
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;
    validateRequirement(id, rows >= 1, 'needs at least one row');
    if (props.mask) {
      validatePackedView(props.mask, ['uint32'], `${id} mask`);
      validateRequirement(id, props.mask.length === rows, 'mask length must equal the row count');
    }
    validatePackedView(props.output.cells, ['uint32x2'], `${id} output.cells`);
    validateRequirement(
      id,
      props.output.cells.length === rows,
      'output.cells length must equal the row count'
    );
    if (props.output.validity) {
      validatePackedView(props.output.validity, ['uint32'], `${id} output.validity`);
      validateRequirement(
        id,
        props.output.validity.length === rows,
        'output.validity length must equal the row count'
      );
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.cells, props.output.validity],
      [props.positions, props.mask]
    );
  }

  /** Returns the single key kernel node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.mask,
      output.cells,
      output.validity
    ]);
    const isUniq = props.scheme === 'uniq';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: `healpix-${isUniq ? 'uniq' : 'nested'}`,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          ...(props.mask
            ? [{name: 'rowMask', view: props.mask, type: 'u32', access: 'read'} as const]
            : []),
          {name: 'cellsOut', view: output.cells, type: 'u32', access: 'read_write'},
          ...(output.validity
            ? [
                {
                  name: 'validityOut',
                  view: output.validity,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: props.positions.length,
        declarations: `const ORDER: u32 = ${props.order}u;
${HEALPIX_INDEX_WGSL}
${IS_FINITE_BITS_WGSL}`,
        body: `let longitude = positions[positionsOffset + 2u * index];
  let latitude = positions[positionsOffset + 2u * index + 1u];
  var isValid = isFiniteBits(longitude) && isFiniteBits(latitude);
  ${props.mask ? 'isValid = isValid && rowMask[rowMaskOffset + index] != 0u;' : ''}
  var key = vec2u(0u);
  if (isValid) {
    key = cellIndexHealpixNestedFromLngLat(longitude, latitude, ORDER);
    ${isUniq ? 'key = cellIndexHealpixUniq(key, ORDER);' : ''}
  }
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  ${output.validity ? 'validityOut[validityOutOffset + index] = select(0u, 1u, isValid);' : ''}`
      })
    ];
  }
}
