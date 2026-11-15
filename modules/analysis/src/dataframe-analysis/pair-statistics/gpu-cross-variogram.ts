// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateGPUStorageBindingByteLength,
  validateMaximumCount,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {PAIR_STATISTICS_FLOAT_WGSL, validatePairStatisticsRowCount} from './pair-statistics-grid';
import {
  PAIR_STATISTICS_PARTIAL_ENTRY_MAXIMUM,
  validatePairStatisticsGridSize
} from './pair-statistics-cell-neighbors';
import {getEmpiricalVariogramNodes} from './empirical-variogram-nodes';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

/** Signed empirical cross-semivariance. Times add a second, temporal lag axis. */
export type GPUCrossVariogramProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  values: GraphDataView<'float32'>;
  /** Defaults to values, giving the ordinary Matheron semivariogram. */
  secondValues?: GraphDataView<'float32'>;
  times?: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** `[maximumDistance, maximumTimeLag]`; positive finite entries. Time ignored without times. */
  parameters: GraphDataView<'float32'>;
  lagCount: number;
  temporalLagCount?: number;
  /**
   * Optional compile-time `[columns, rows]` cell lattice. Each encoding fits it to the bounds of the
   * included rows with cells at least `maximumDistance` wide, so pairs come from the 3 x 3 cells
   * around each row instead of every row, and up to `2^24 / bins` rows are accepted. Pair counts
   * are unchanged; neighboring cell ranges are merged in input-row order before summation, so
   * lattice dimensions do not change pair-sum order. Omitted uses a 256 by 256 lattice with
   * the same row and partial-table limits as an explicit lattice.
   */
  gridSize?: readonly [number, number];
  output: {
    /** Spatial-major bins: spatialLag * temporalLagCount + temporalLag. */
    semivariances: GraphDataView<'float32'>;
    /** Unordered pairs per bin, clamped to 0xffffffff on overflow (status 3, NaN estimate). */
    pairCounts: GraphDataView<'uint32'>;
    /** Per bin: 0 valid, 1 empty/invalid parameters, 2 nonfinite arithmetic, 3 pair-count overflow. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Deterministic GPU pair accumulation of (x_i-x_j)(y_i-y_j)/2. Negative cross-semivariances are
 * retained. Unordered pairs counted once; bins are left-closed and right-open, except the final
 * edge is included. Complete-case masking includes both variables, coordinates and optional time.
 * Forwards through the shared empirical variogram front end, adapting the legacy parameter layout
 * and keeping zero lags in the first bin. Legacy accumulation uses raw value differences and
 * indexed per-row floating sums with bounded tree reductions; it does not center or quantize values.
 * The default lattice is 256 by 256; unrepresentable extents collapse it to one cell, keeping
 * finite-distance pairs. Merging sorted cell ranges preserves input-row order within every pair
 * sum. The fixed tree over row partials can round differently from a serial finishing sum.
 * No edge correction or implicit projection.
 *
 * @deprecated Use `GPUSpaceTimeVariogram`, whose explicit zero-lag classes separate coincident
 * pairs from positive lags. This forwarding class preserves the original props and outputs.
 */
export class GPUCrossVariogram implements GPUCommandNodeProducer {
  readonly id: string;
  readonly spatialContext: GPUSpatialContext;
  constructor(readonly props: GPUCrossVariogramProps) {
    this.id = props.id ?? 'cross-variogram';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const rows = props.positions.length;
    const temporalCount = props.temporalLagCount ?? 1;
    if (
      !Number.isInteger(props.lagCount) ||
      props.lagCount < 1 ||
      !Number.isInteger(temporalCount) ||
      temporalCount < 1 ||
      props.lagCount * temporalCount > 128 ||
      (!props.times && temporalCount !== 1)
    )
      throw new Error(`${this.id}: invalid bounded pair shape`);
    validateCapacity(this.id, 'positions length', rows, 1, 2 ** 24 - 1);
    validatePairStatisticsGridSize(this.id, props.gridSize ?? [256, 256]);
    validateMaximumCount(
      this.id,
      'positions length x bins',
      rows * props.lagCount * temporalCount,
      PAIR_STATISTICS_PARTIAL_ENTRY_MAXIMUM
    );
    validateSinglePackedView(props.positions, ['float32x2'], this.id);
    validateSinglePackedView(props.parameters, ['float32'], this.id);
    if (props.parameters.length < (props.times ? 2 : 1))
      throw new Error(`${this.id}: missing lag parameters`);
    for (const view of [props.values, props.secondValues, props.times])
      if (view) {
        validateSinglePackedView(view, ['float32'], this.id);
        if (view.length !== rows) throw new Error(`${this.id}: values must match positions`);
      }
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], this.id);
      if (props.mask.length !== rows) throw new Error(`${this.id}: mask length mismatch`);
    }
    const bins = props.lagCount * temporalCount;
    for (const [name, view] of Object.entries(props.output)) {
      validateSinglePackedView(view, [name === 'semivariances' ? 'float32' : 'uint32'], this.id);
      if (view.length !== bins) throw new Error(`${this.id}: output length must match bins`);
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      Object.values(props.output),
      getBindings(props).map(binding => binding.view)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const bindings = getBindings(props);
    validateGraphViewsBelongToGraph(id, graph, [
      ...bindings.map(binding => binding.view),
      ...Object.values(props.output)
    ]);
    return getLegacyVariogramNodes(graph, id, props, props.gridSize ?? [256, 256]);
  }
}
export {GPUSpaceTimeVariogram} from './gpu-space-time-variogram';
export type {GPUSpaceTimeVariogramProps} from './gpu-space-time-variogram';

function getBindings(props: GPUCrossVariogramProps): WGSLKernelBinding[] {
  return [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'values', view: props.values, type: 'f32', access: 'read'},
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
    ...(props.secondValues
      ? [
          {
            name: 'secondValues',
            view: props.secondValues,
            type: 'f32' as const,
            access: 'read' as const
          }
        ]
      : []),
    ...(props.times
      ? [{name: 'times', view: props.times, type: 'f32' as const, access: 'read' as const}]
      : []),
    ...(props.mask
      ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
      : [])
  ];
}

/** Adapt complete-case bounds and the legacy lag parameters, then forward to shared kernels. */
function getLegacyVariogramNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUCrossVariogramProps,
  gridSize: readonly [number, number]
): GPUCommandNode<Parameters>[] {
  validatePairStatisticsRowCount(id, graph.device, props.positions.length);
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'legacy variogram partials',
    props.positions.length * props.lagCount * (props.temporalLagCount ?? 1) * 4,
    'reduce rows or lag counts'
  );
  const inclusion = createTransientView(graph, `${id}-inclusion`, 'uint32', props.positions.length);
  const boundKeys = createTransientView(graph, `${id}-bound-keys`, 'uint32', 4);
  const gridParameters = createTransientView(graph, `${id}-grid-parameters`, 'float32', 8);
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-bounds-clear`,
      operation: 'GPUCrossVariogram',
      variant: 'bounds-clear',
      invocationCount: 4,
      bindings: [{name: 'boundKeys', view: boundKeys, type: 'u32', access: 'read_write'}],
      body: 'boundKeys[boundKeysOffset + index] = select(0u, 0xffffffffu, index < 2u);'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-inclusion`,
      operation: 'GPUCrossVariogram',
      variant: 'inclusion',
      invocationCount: props.positions.length,
      bindings: [
        ...getBindings(props).filter(binding => binding.name !== 'parameters'),
        {name: 'inclusion', view: inclusion, type: 'u32', access: 'read_write'},
        {name: 'boundKeys', view: boundKeys, type: 'atomic<u32>', access: 'read_write'}
      ],
      declarations: `${PAIR_STATISTICS_FLOAT_WGSL}\n${IS_FINITE_BITS_WGSL}`,
      body: `let x = positions[positionsOffset + 2u * index];
let y = positions[positionsOffset + 2u * index + 1u];
let value = values[valuesOffset + index];
let secondValue = ${props.secondValues ? 'secondValues[secondValuesOffset + index]' : 'value'};
let time = ${props.times ? 'times[timesOffset + index]' : '0.0'};
let included = ${props.mask ? 'mask[maskOffset + index] != 0u &&' : ''} isFiniteBits(x) && isFiniteBits(y) && isFiniteBits(value) && isFiniteBits(secondValue) && isFiniteBits(time);
inclusion[inclusionOffset + index] = select(0u, 1u, included);
if (included) {
  atomicMin(&boundKeys[boundKeysOffset], getOrderedFloatKey(x));
  atomicMin(&boundKeys[boundKeysOffset + 1u], getOrderedFloatKey(y));
  atomicMax(&boundKeys[boundKeysOffset + 2u], getOrderedFloatKey(x));
  atomicMax(&boundKeys[boundKeysOffset + 3u], getOrderedFloatKey(y));
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-grid-parameters`,
      operation: 'GPUCrossVariogram',
      variant: 'grid-parameters',
      invocationCount: 8,
      bindings: [
        {name: 'boundKeys', view: boundKeys, type: 'u32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'gridParameters', view: gridParameters, type: 'f32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${PAIR_STATISTICS_FLOAT_WGSL}`,
      body: `let maximumDistance = parameters[parametersOffset];
let maximumTime = ${props.times ? 'parameters[parametersOffset + 1u]' : '1.0'};
var value = 0.0;
if (index < 4u) { value = getOrderedFloatValue(boundKeys[boundKeysOffset + index]); }
if (index == 4u) { value = select(getNaN(), maximumDistance, maximumTime > 0.0 && isFiniteBits(maximumTime)); }
if (index == 5u) { value = maximumTime; }
gridParameters[gridParametersOffset + index] = value;`
    }),
    ...getEmpiricalVariogramNodes(
      graph,
      `${id}-empirical`,
      {
        ...props,
        parameters: gridParameters,
        mask: inclusion,
        gridSize,
        temporalLagCount: props.temporalLagCount ?? 1
      },
      true
    )
  ];
}
