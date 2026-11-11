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
  validateMaximumCount,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPairStatisticsInputNodes,
  getPairStatisticsSharedWGSL,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsRowCount
} from './pair-statistics-grid';
import {
  getPairStatisticsCellNeighborLoopWGSL,
  getPairStatisticsSegmentTotalNodes,
  PAIR_STATISTICS_PARTIAL_ENTRY_MAXIMUM,
  validatePairStatisticsGridSize
} from './pair-statistics-cell-neighbors';
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
   * are unchanged; sums are added in cell order, so semivariances can differ in the last float32
   * bits. Omitted keeps the dense all-pairs traversal (at most 16,384 rows).
   */
  gridSize?: readonly [number, number];
  output: {
    /** Spatial-major bins: spatialLag * temporalLagCount + temporalLag. */
    semivariances: GraphDataView<'float32'>;
    pairCounts: GraphDataView<'uint32'>;
    /** Per bin: 0 valid, 1 empty/invalid parameters, 2 nonfinite accumulated arithmetic. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Deterministic GPU pair accumulation of (x_i-x_j)(y_i-y_j)/2. Negative cross-semivariances are
 * retained. Unordered pairs counted once; bins are left-closed and right-open, except the final
 * edge is included. Complete-case masking includes both variables, coordinates and optional time.
 * Without `gridSize`, uses bounded dense traversal (at most 16,384 rows and 128 bins), O(rows² *
 * bins) work and O(rows * bins) scratch. With `gridSize`, pairs come from a cell index whose cells
 * are at least `maximumDistance` wide. No edge correction or implicit projection.
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
    validateCapacity(this.id, 'positions length', rows, 1, props.gridSize ? 2 ** 24 - 1 : 16384);
    if (props.gridSize) {
      validatePairStatisticsGridSize(this.id, props.gridSize);
      validateMaximumCount(
        this.id,
        'positions length x bins',
        rows * props.lagCount * temporalCount,
        PAIR_STATISTICS_PARTIAL_ENTRY_MAXIMUM
      );
    }
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
    const rows = props.positions.length;
    const temporalCount = props.temporalLagCount ?? 1;
    const bins = props.lagCount * temporalCount;
    const bindings = getBindings(props);
    validateGraphViewsBelongToGraph(id, graph, [
      ...bindings.map(binding => binding.view),
      ...Object.values(props.output)
    ]);
    if (props.gridSize) {
      return getGridNodes(graph, id, props, props.gridSize);
    }
    const partials = createTransientView(graph, `${id}-partials`, 'float32', rows * bins * 2);
    const pairNodes = [
      createWGSLKernelNode(graph, {
        id: `${id}-pairs`,
        operation: 'GPUCrossVariogram',
        variant: props.times ? 'space-time' : 'cross',
        invocationCount: rows * bins,
        bindings: [
          ...bindings,
          {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
fn included(row: u32) -> bool {
  ${props.mask ? 'if (mask[maskOffset + row] == 0u) { return false; }' : ''}
  return isFiniteBits(positions[positionsOffset + 2u * row]) && isFiniteBits(positions[positionsOffset + 2u * row + 1u]) && isFiniteBits(values[valuesOffset + row])
    ${props.secondValues ? '&& isFiniteBits(secondValues[secondValuesOffset + row])' : ''}
    ${props.times ? '&& isFiniteBits(times[timesOffset + row])' : ''};
}`,
        body: `let focus = index / ${bins}u; let bin = index % ${bins}u;
let maximumDistance = parameters[parametersOffset];
let maximumTime = ${props.times ? 'parameters[parametersOffset + 1u]' : '1.0'};
var sum = 0.0; var count = 0.0;
if (included(focus) && maximumDistance > 0.0 && isFiniteBits(maximumDistance) && maximumTime > 0.0 && isFiniteBits(maximumTime)) {
  for (var neighbor = focus + 1u; neighbor < ${rows}u; neighbor++) {
    if (!included(neighbor)) { continue; }
    let delta = vec2<f32>(positions[positionsOffset + 2u * focus] - positions[positionsOffset + 2u * neighbor], positions[positionsOffset + 2u * focus + 1u] - positions[positionsOffset + 2u * neighbor + 1u]);
    let distance = length(delta);
    let time = ${props.times ? 'abs(times[timesOffset + focus] - times[timesOffset + neighbor])' : '0.0'};
    if (!isFiniteBits(distance) || distance > maximumDistance || time > maximumTime) { continue; }
    let spatialBin = min(u32(distance / maximumDistance * ${props.lagCount}.0), ${props.lagCount - 1}u);
    let temporalBin = min(u32(time / maximumTime * ${temporalCount}.0), ${temporalCount - 1}u);
    if (spatialBin * ${temporalCount}u + temporalBin != bin) { continue; }
    let difference = values[valuesOffset + focus] - values[valuesOffset + neighbor];
    let secondDifference = ${props.secondValues ? 'secondValues[secondValuesOffset + focus] - secondValues[secondValuesOffset + neighbor]' : 'difference'};
    sum += 0.5 * difference * secondDifference; count += 1.0;
  }
}
partials[partialsOffset + index * 2u] = sum; partials[partialsOffset + index * 2u + 1u] = count;`
      })
    ];
    return [
      ...pairNodes,
      createWGSLKernelNode(graph, {
        id: `${id}-finish`,
        operation: 'GPUCrossVariogram',
        variant: 'finish',
        invocationCount: bins,
        bindings: [
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'result', view: props.output.semivariances, type: 'f32', access: 'read_write'},
          {name: 'counts', view: props.output.pairCounts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `var sum = 0.0; var count = 0u;
for (var row = 0u; row < ${rows}u; row++) { let base = partialsOffset + (row * ${bins}u + index) * 2u; sum += partials[base]; count += u32(partials[base + 1u]); }
${CROSS_VARIOGRAM_FINISH_WGSL}`
      })
    ];
  }
}

/**
 * Grid path: complete-case rows and their bounds, a cell index fitted to those bounds with cells
 * at least `maximumDistance` wide, then one invocation per focus row that visits the 3 x 3
 * neighboring cells and writes its `(sum, count)` per bin into bin-major tables. Segment totals
 * (at most 256 entries per invocation) reduce the tables per bin, and the finish pass divides; no
 * pass loops over every row in one invocation.
 */
function getGridNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUCrossVariogramProps,
  gridSize: readonly [number, number]
): GPUCommandNode<Parameters>[] {
  validatePairStatisticsRowCount(id, graph.device, props.positions.length);
  const rows = props.positions.length;
  const temporalCount = props.temporalLagCount ?? 1;
  const bins = props.lagCount * temporalCount;
  // Bin-major per-focus partials and their per-bin totals.
  const binSums = createTransientView(graph, `${id}-bin-sums`, 'float32', bins * rows);
  const binCounts = createTransientView(graph, `${id}-bin-counts`, 'uint32', bins * rows);
  const totalSums = createTransientView(graph, `${id}-total-sums`, 'float32', bins);
  const totalCounts = createTransientView(graph, `${id}-total-counts`, 'uint32', bins);
  const inclusion = createTransientView(graph, `${id}-inclusion`, 'uint32', rows);
  // Per row: value, second value (or value) and time (or 0), read by the pair pass in one binding.
  const attributes = createTransientView(graph, `${id}-attributes`, 'float32', rows * 3);
  const boundKeys = createTransientView(graph, `${id}-bound-keys`, 'uint32', 4);
  // [minimumX, minimumY, maximumX, maximumY, maximumDistance, maximumTime]
  const gridParameters = createTransientView(graph, `${id}-grid-parameters`, 'float32', 6);
  const sharedWGSL = getPairStatisticsSharedWGSL(gridSize);
  const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
    id: `${id}-grid`,
    operation: 'GPUCrossVariogram',
    positions: props.positions,
    parameters: gridParameters,
    gridSize,
    mask: inclusion
  });
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-bounds-clear`,
      operation: 'GPUCrossVariogram',
      variant: 'bounds-clear',
      invocationCount: 4,
      bindings: [{name: 'boundKeys', view: boundKeys, type: 'u32', access: 'read_write'}],
      body: 'boundKeys[boundKeysOffset + index] = select(0u, 0xffffffffu, index < 2u);'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-inclusion`,
      operation: 'GPUCrossVariogram',
      variant: 'inclusion',
      invocationCount: rows,
      bindings: [
        ...getBindings(props).filter(binding => binding.name !== 'parameters'),
        {name: 'inclusion', view: inclusion, type: 'u32', access: 'read_write'},
        {name: 'attributes', view: attributes, type: 'f32', access: 'read_write'},
        {name: 'boundKeys', view: boundKeys, type: 'atomic<u32>', access: 'read_write'}
      ],
      declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${IS_FINITE_BITS_WGSL}`,
      body: `let x = positions[positionsOffset + 2u * index]; let y = positions[positionsOffset + 2u * index + 1u];
let value = values[valuesOffset + index];
let secondValue = ${props.secondValues ? 'secondValues[secondValuesOffset + index]' : 'value'};
let time = ${props.times ? 'times[timesOffset + index]' : '0.0'};
let included = ${props.mask ? 'mask[maskOffset + index] != 0u &&' : ''} isFiniteBits(x) && isFiniteBits(y) && isFiniteBits(value) && isFiniteBits(secondValue) && isFiniteBits(time);
inclusion[inclusionOffset + index] = select(0u, 1u, included);
attributes[attributesOffset + index * 3u] = value; attributes[attributesOffset + index * 3u + 1u] = secondValue; attributes[attributesOffset + index * 3u + 2u] = time;
if (included) {
  atomicMin(&boundKeys[boundKeysOffset], getOrderedFloatKey(x)); atomicMin(&boundKeys[boundKeysOffset + 1u], getOrderedFloatKey(y));
  atomicMax(&boundKeys[boundKeysOffset + 2u], getOrderedFloatKey(x)); atomicMax(&boundKeys[boundKeysOffset + 3u], getOrderedFloatKey(y));
}`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-grid-parameters`,
      operation: 'GPUCrossVariogram',
      variant: 'grid-parameters',
      invocationCount: 1,
      bindings: [
        {name: 'boundKeys', view: boundKeys, type: 'u32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'gridParameters', view: gridParameters, type: 'f32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}
${IS_FINITE_BITS_WGSL}
${PAIR_STATISTICS_FLOAT_WGSL}`,
      // No included row leaves the minimum keys at 0xffffffff, which decode to NaN bounds, and an
      // unusable maximum time gives a NaN distance: either way the lattice excludes every row.
      body: `let maximumDistance = parameters[parametersOffset];
let maximumTime = ${props.times ? 'parameters[parametersOffset + 1u]' : '1.0'};
for (var slot = 0u; slot < 4u; slot++) { gridParameters[gridParametersOffset + slot] = getOrderedFloatValue(boundKeys[boundKeysOffset + slot]); }
gridParameters[gridParametersOffset + 4u] = select(getNaN(), maximumDistance, maximumTime > 0.0 && isFiniteBits(maximumTime));
gridParameters[gridParametersOffset + 5u] = maximumTime;`
    }),
    ...inputs.nodes,
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-grid-pairs`,
      operation: 'GPUCrossVariogram',
      variant: props.times ? 'grid-space-time' : 'grid-cross',
      invocationCount: rows,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'attributes', view: attributes, type: 'f32', access: 'read'},
        {name: 'parameters', view: gridParameters, type: 'f32', access: 'read'},
        {name: 'cellKeys', view: inputs.cellKeys, type: 'u32', access: 'read'},
        {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
        {name: 'sortedRows', view: inputs.sortedRows, type: 'u32', access: 'read'},
        {name: 'binSums', view: binSums, type: 'f32', access: 'read_write'},
        {name: 'binCounts', view: binCounts, type: 'u32', access: 'read_write'}
      ],
      declarations: `${sharedWGSL}
${IS_FINITE_BITS_WGSL}
const BINS: u32 = ${bins}u;`,
      body: `var sums: array<f32, BINS>; var counts: array<u32, BINS>;
let lattice = readLattice();
let focus = index; let focusKey = cellKeys[cellKeysOffset + focus];
let maximumDistance = parameters[parametersOffset + 4u]; let maximumTime = parameters[parametersOffset + 5u];
if (lattice.valid && focusKey < CELL_COUNT) ${getPairStatisticsCellNeighborLoopWGSL(
        'focusKey',
        `if (neighbor <= focus) { continue; }
    let delta = vec2<f32>(positions[positionsOffset + 2u * focus] - positions[positionsOffset + 2u * neighbor], positions[positionsOffset + 2u * focus + 1u] - positions[positionsOffset + 2u * neighbor + 1u]);
    let distance = length(delta);
    let time = abs(attributes[attributesOffset + focus * 3u + 2u] - attributes[attributesOffset + neighbor * 3u + 2u]);
    if (!isFiniteBits(distance) || distance > maximumDistance || time > maximumTime) { continue; }
    let spatialBin = min(u32(distance / maximumDistance * ${props.lagCount}.0), ${props.lagCount - 1}u);
    let temporalBin = min(u32(time / maximumTime * ${temporalCount}.0), ${temporalCount - 1}u);
    let bin = spatialBin * ${temporalCount}u + temporalBin;
    let difference = attributes[attributesOffset + focus * 3u] - attributes[attributesOffset + neighbor * 3u];
    let secondDifference = attributes[attributesOffset + focus * 3u + 1u] - attributes[attributesOffset + neighbor * 3u + 1u];
    sums[bin] += 0.5 * difference * secondDifference; counts[bin] += 1u;`
      )}
for (var bin = 0u; bin < BINS; bin++) { binSums[binSumsOffset + bin * ${rows}u + focus] = sums[bin]; binCounts[binCountsOffset + bin * ${rows}u + focus] = counts[bin]; }`
    }),
    ...getPairStatisticsSegmentTotalNodes<Parameters>(graph, {
      id: `${id}-total-sums`,
      operation: 'GPUCrossVariogram',
      segmentCount: bins,
      rowCount: rows,
      input: binSums,
      output: totalSums
    }),
    ...getPairStatisticsSegmentTotalNodes<Parameters>(graph, {
      id: `${id}-total-counts`,
      operation: 'GPUCrossVariogram',
      segmentCount: bins,
      rowCount: rows,
      input: binCounts,
      output: totalCounts
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-finish`,
      operation: 'GPUCrossVariogram',
      variant: 'grid-finish',
      invocationCount: bins,
      bindings: [
        {name: 'totalSums', view: totalSums, type: 'f32', access: 'read'},
        {name: 'totalCounts', view: totalCounts, type: 'u32', access: 'read'},
        {name: 'result', view: props.output.semivariances, type: 'f32', access: 'read_write'},
        {name: 'counts', view: props.output.pairCounts, type: 'u32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
      body: `let sum = totalSums[totalSumsOffset + index]; let count = totalCounts[totalCountsOffset + index];
${CROSS_VARIOGRAM_FINISH_WGSL}`
    })
  ];
}

/** Writes one bin's semivariance, count and status from its `sum` and `count`. */
const CROSS_VARIOGRAM_FINISH_WGSL = `let value = sum / f32(count);
result[resultOffset + index] = select(getNaN(), value, count > 0u && isFiniteBits(value));
counts[countsOffset + index] = count;
status[statusOffset + index] = select(select(2u, 0u, isFiniteBits(value)), 1u, count == 0u);`;

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
