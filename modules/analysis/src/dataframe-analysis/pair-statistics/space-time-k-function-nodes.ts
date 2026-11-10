// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
import {
  getStorageBindingElementLimit,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  DECODE_ORDERED_KEY_WGSL,
  GET_ORDERED_KEY_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  getPairHistogramNodes,
  getPairHistogramReadWGSL,
  PAIR_HISTOGRAM_LARGE_TERM_SHIFT
} from './pair-histogram';
import {
  getGPUPairStatisticsMaximumRowCount,
  getPairStatisticsInputNodes,
  PAIR_STATISTICS_FLOAT_WGSL,
  type PairStatisticsInputs
} from './pair-statistics-grid';
import {
  GPU_SPACE_TIME_K_WEIGHT_CAP,
  SPACE_TIME_K_EDGE_WEIGHT_WGSL,
  SPACE_TIME_K_WEIGHT_SCALE
} from './space-time-k-function-wgsl';

/** Edge correction of the space-time K family. See {@link GPUSpaceTimeKFunction}. */
export type GPUSpaceTimeKEdgeCorrection = 'none' | 'isotropic';

/** Kernel of the space-time pair correlation along one axis. */
export type GPUSpaceTimePairCorrelationKernel = 'box' | 'epanechnikov';

/** Number of float32 elements in a space-time K parameter view. */
export const GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH = 11;

/** Largest number of spatial classes or of time classes. */
export const GPU_SPACE_TIME_K_MAXIMUM_CLASS_COUNT = 64;

/** Largest number of `(distance, time)` classes per event of the local (LISTA) output. */
export const GPU_SPACE_TIME_K_MAXIMUM_LOCAL_BINS = 64;

/** Bytes of one row of the row table of {@link createSpaceTimeKNodes}: four float32. */
const ROW_TABLE_BYTES = 16;

/**
 * Largest row count of the space-time K family on `device`: the row table holds
 * {@link ROW_TABLE_BYTES} bytes per row in one storage binding, which is 8388608 rows on a default
 * device (128 MiB bindings), below the 11184810 of the plain pair statistics.
 */
export function getGPUSpaceTimeKMaximumRowCount(device: Device): number {
  return Math.min(
    getGPUPairStatisticsMaximumRowCount(device),
    getStorageBindingElementLimit(device, ROW_TABLE_BYTES)
  );
}

/** Throws unless `rows` is at most {@link getGPUSpaceTimeKMaximumRowCount}. @internal */
export function validateSpaceTimeKRowCount(id: string, device: Device, rows: number): void {
  validateMaximumCount(id, 'positions length', rows, getGPUSpaceTimeKMaximumRowCount(device));
}

/** Per-frame settings of the space-time K family, packed by {@link getGPUSpaceTimeKFunctionParameterValues}. */
export type GPUSpaceTimeKFunctionSettings = {
  /** Spatial study window `[minX, minY, maxX, maxY]`, a rectangle with positive area. Rows outside are excluded. */
  bounds: readonly [number, number, number, number];
  /** Largest distance `u` of the K grid, in position units. Class `a` has radius `u * (a + 1) / distanceClassCount`. */
  maximumDistance: number;
  /** Largest time lag `v` of the K grid, in the unit of `times`. Class `b` has lag `v * (b + 1) / timeClassCount`. */
  maximumTimeLag: number;
  /**
   * Observation interval `[start, end]` of the times, `start < end`. Rows whose time is outside it
   * are excluded. Its length is `|T|`.
   */
  timeWindow: readonly [number, number];
  /** Spatial bandwidth `hs` of the pair correlation kernel, positive. Needed with `output.pairCorrelation`. */
  spatialBandwidth?: number;
  /** Temporal bandwidth `ht` of the pair correlation kernel, positive. Needed with `output.pairCorrelation`. */
  temporalBandwidth?: number;
};

/**
 * Packs the 11-element float32 layout `[minX, minY, maxX, maxY, searchDistance, maximumDistance,
 * maximumTimeLag, timeStart, timeEnd, spatialBandwidth, temporalBandwidth]`, where
 * `searchDistance = maximumDistance + spatialBandwidth` (the pair search radius of the shared
 * pair histogram).
 *
 * @throws If a value is not finite, the bounds or time window are inverted or empty, a maximum is
 * not positive, or a bandwidth is given and is not positive.
 */
export function getGPUSpaceTimeKFunctionParameterValues(
  settings: GPUSpaceTimeKFunctionSettings,
  target: Float32Array = new Float32Array(GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH)
): Float32Array {
  const name = 'getGPUSpaceTimeKFunctionParameterValues';
  if (target.length < GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH) {
    throw new Error(
      `${name} target must hold ${GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH} elements`
    );
  }
  const {bounds, timeWindow, maximumDistance, maximumTimeLag} = settings;
  if (bounds.length !== 4 || !bounds.every(Number.isFinite)) {
    throw new Error(`${name} bounds must be four finite numbers [minX, minY, maxX, maxY]`);
  }
  if (!(bounds[0] < bounds[2] && bounds[1] < bounds[3])) {
    throw new Error(`${name} bounds must have positive area`);
  }
  if (
    timeWindow.length !== 2 ||
    !timeWindow.every(Number.isFinite) ||
    !(timeWindow[0] < timeWindow[1])
  ) {
    throw new Error(`${name} timeWindow must be finite [start, end] with start < end`);
  }
  for (const [label, value] of [
    ['maximumDistance', maximumDistance],
    ['maximumTimeLag', maximumTimeLag]
  ] as const) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`${name} ${label} must be positive and finite`);
    }
  }
  const spatialBandwidth = settings.spatialBandwidth ?? 0;
  const temporalBandwidth = settings.temporalBandwidth ?? 0;
  for (const [label, value] of [
    ['spatialBandwidth', spatialBandwidth],
    ['temporalBandwidth', temporalBandwidth]
  ] as const) {
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(`${name} ${label} must be non-negative and finite`);
    }
  }
  target.set([
    ...bounds,
    maximumDistance + spatialBandwidth,
    maximumDistance,
    maximumTimeLag,
    ...timeWindow,
    spatialBandwidth,
    temporalBandwidth
  ]);
  return target;
}

/** Views and shape shared by the nodes of the space-time K family. @internal */
export type SpaceTimeKConfig = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  times: GraphDataView<'float32'>;
  intensities?: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
  gridSize: readonly [number, number];
  distanceClassCount: number;
  timeClassCount: number;
  edgeCorrection: GPUSpaceTimeKEdgeCorrection;
  pairCorrelationKernel?: GPUSpaceTimePairCorrelationKernel;
  /** Cumulative `K(u_a, v_b)`, `a * timeClassCount + b`. */
  k: GraphDataView<'float32'>;
  /** Spatial `K_s(u_a)`; built when present. Homogeneous only. */
  spatialK?: GraphDataView<'float32'>;
  /** Temporal `K_t(v_b)`; built when present. Homogeneous only. */
  temporalK?: GraphDataView<'float32'>;
  /** `D = K - K_s K_t`; needs `spatialK` and `temporalK`. */
  d?: GraphDataView<'float32'>;
  pairCorrelation?: GraphDataView<'float32'>;
  pairCounts?: GraphDataView<'uint32'>;
  /** Local `K_i(u_a, v_b)`, `row * bins + a * timeClassCount + b`; built when present. */
  local?: GraphDataView<'float32'>;
  status?: GraphDataView<'uint32'>;
};

/** Nodes and shared views produced by {@link createSpaceTimeKNodes}. @internal */
export type SpaceTimeKNodes<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  inputs: PairStatisticsInputs<Parameters>;
  /** Four floats per row: time, `1 / lambda'`, temporal edge radius, unused. */
  rowTable: GraphDataView<'float32'>;
  /** One uint32 per row: 1 when the row passed the row filter (time, intensity and mask). */
  effectiveMask: GraphDataView<'uint32'>;
};

/** Bit weight of one large-channel unit, see `PAIR_HISTOGRAM_LARGE_TERM_SHIFT`. */
const LARGE_UNIT = 2 ** PAIR_HISTOGRAM_LARGE_TERM_SHIFT;

/**
 * Returns the WGSL constants shared by the kernels of the family. `BINS` is the number of
 * `(distance, time)` classes.
 */
function getConstantsWGSL(config: SpaceTimeKConfig): string {
  const {distanceClassCount, timeClassCount} = config;
  return /* wgsl */ `
const DISTANCE_CLASS_COUNT: u32 = ${distanceClassCount}u;
const TIME_CLASS_COUNT: u32 = ${timeClassCount}u;
const BINS: u32 = ${distanceClassCount * timeClassCount}u;
const WEIGHT_SCALE: f32 = ${SPACE_TIME_K_WEIGHT_SCALE}.0;
const WEIGHT_CAP: f32 = ${GPU_SPACE_TIME_K_WEIGHT_CAP}.0;
const LARGE_UNIT: f32 = ${LARGE_UNIT}.0;
const EDGE_ISOTROPIC: bool = ${config.edgeCorrection === 'isotropic'};
const INHOMOGENEOUS: bool = ${Boolean(config.intensities)};
`;
}

/**
 * Builds the observed-pattern pass of {@link GPUSpaceTimeKFunction}: the row filter, the cell
 * index, one indexed pair histogram that fills the `(distance, time)` classes (and the spatial
 * marginal, the pair correlation and the per-event local sums), the temporal marginal on the
 * sorted times, and the finishing kernels. Everything is exact integer accumulation except the
 * per-event local sums, which add in the fixed neighbor order.
 * @internal
 */
export function createSpaceTimeKNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  config: SpaceTimeKConfig
): SpaceTimeKNodes<Parameters> {
  const {id, operation, positions, times, parameters, gridSize} = config;
  const rows = positions.length;
  const U = config.distanceClassCount;
  const V = config.timeClassCount;
  const bins = U * V;
  const cellCount = gridSize[0] * gridSize[1];
  const constants = getConstantsWGSL(config);
  const hasMarginal = Boolean(config.spatialK);
  const hasTemporal = Boolean(config.temporalK);
  const hasPairCorrelation = Boolean(config.pairCorrelation);
  const hasLocal = Boolean(config.local);
  const nodes: GPUCommandNode<Parameters>[] = [];

  // Row filter: finite time inside the observation interval, finite positive intensity, mask.
  const effectiveMask = createTransientView(graph, `${id}-effective-mask`, 'uint32', rows);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-effective-mask`,
      operation,
      variant: `effective-mask-${config.mask ? 'mask' : 'all'}-${config.intensities ? 'intensity' : 'homogeneous'}`,
      bindings: [
        {name: 'times', view: times, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        ...(config.mask
          ? [{name: 'mask', view: config.mask, type: 'u32', access: 'read'} as const]
          : []),
        ...(config.intensities
          ? [{name: 'intensities', view: config.intensities, type: 'f32', access: 'read'} as const]
          : []),
        {name: 'effectiveMask', view: effectiveMask, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: PAIR_STATISTICS_FLOAT_WGSL,
      body: `let timeStart = parameters[parametersOffset + 7u];
  let timeEnd = parameters[parametersOffset + 8u];
  let time = times[timesOffset + index];
  var keep = isFiniteFloat(timeStart) && isFiniteFloat(timeEnd) && timeStart < timeEnd &&
    isFiniteFloat(time) && time >= timeStart && time <= timeEnd;
  ${config.mask ? 'keep = keep && mask[maskOffset + index] != 0u;' : ''}
  ${config.intensities ? 'let intensity = intensities[intensitiesOffset + index];\n  keep = keep && isFiniteFloat(intensity) && intensity > 0.0;' : ''}
  effectiveMask[effectiveMaskOffset + index] = select(0u, 1u, keep);`
    })
  );
  const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
    id,
    operation,
    positions,
    parameters,
    gridSize,
    mask: effectiveMask
  });
  nodes.push(...inputs.nodes);

  // Per row: time, 1 / lambda', and the radius of the temporal isotropic weight.
  const rowTable = createTransientView(graph, `${id}-row-table`, 'float32', 4 * rows);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-row-table`,
      operation,
      variant: `row-table-${config.intensities ? 'intensity' : 'homogeneous'}`,
      bindings: [
        {name: 'times', view: times, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
        ...(config.intensities
          ? [{name: 'intensities', view: config.intensities, type: 'f32', access: 'read'} as const]
          : []),
        {name: 'rowTable', view: rowTable, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: `const CELL_COUNT: u32 = ${cellCount}u;`,
      body: `let timeStart = parameters[parametersOffset + 7u];
  let timeEnd = parameters[parametersOffset + 8u];
  let time = times[timesOffset + index];
  ${
    config.intensities
      ? `// lambda' = lambda / (n / (|W| |T|)), so 1 / lambda' = lambdaBar / lambda.
  let area = (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]);
  let lambdaBar = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]) / (area * (timeEnd - timeStart));
  let inverse = lambdaBar / intensities[intensitiesOffset + index];`
      : 'let inverse = 1.0;'
  }
  rowTable[rowTableOffset + 4u * index] = time;
  rowTable[rowTableOffset + 4u * index + 1u] = inverse;
  rowTable[rowTableOffset + 4u * index + 2u] = min(time - timeStart, timeEnd - time);
  rowTable[rowTableOffset + 4u * index + 3u] = 0.0;`
    })
  );

  if (config.local) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-local-fill`,
        operation,
        variant: 'local-fill',
        bindings: [{name: 'local', view: config.local, type: 'f32', access: 'read_write'}],
        invocationCount: config.local.length,
        declarations: PAIR_STATISTICS_FLOAT_WGSL,
        body: 'local[localOffset + index] = getQuietNaN(index);'
      })
    );
  }

  // The pair histogram. Slots: the `bins` classes, then the `U` spatial-marginal classes.
  const slotCount = bins + (hasMarginal ? U : 0);
  const channelCount = hasPairCorrelation ? 5 : 3;
  const kernelName = config.pairCorrelationKernel ?? 'box';
  const getKernelWGSL = (axis: 'Spatial' | 'Temporal') => `
fn get${axis}PairKernel(offset: f32, bandwidth: f32) -> f32 {
  // The normalised kernel times its bandwidth, so the accumulated term stays of order one.
  let scaled = offset / bandwidth;
  if (!(abs(scaled) <= 1.0)) { return 0.0; }
  return ${kernelName === 'box' ? '0.5' : '0.75 * (1.0 - scaled * scaled)'};
}`;
  const histogram = getPairHistogramNodes<Parameters>(graph, {
    id: `${id}-histogram`,
    operation,
    positions,
    parameters,
    gridSize,
    sortedRows: inputs.sortedRows,
    cellOffsets: inputs.cellOffsets,
    slotCount,
    channelCount,
    pairOrder: 'ordered',
    extraBindings: [
      {name: 'rowTable', view: rowTable, type: 'f32', access: 'read'},
      ...(config.local
        ? [{name: 'local', view: config.local, type: 'f32', access: 'read_write'} as const]
        : [])
    ] satisfies WGSLKernelBinding[],
    declarations: `${constants}
${SPACE_TIME_K_EDGE_WEIGHT_WGSL}
const HAS_MARGINAL: bool = ${hasMarginal};
const HAS_PAIR_CORRELATION: bool = ${hasPairCorrelation};
const HAS_LOCAL: bool = ${hasLocal};
${getKernelWGSL('Spatial')}
${getKernelWGSL('Temporal')}`,
    focusPrologue: `let focusBase = rowTableOffset + focus * 4u;
    let focusTime = rowTable[focusBase];
    let focusInverse = rowTable[focusBase + 1u];
    let focusRadius = rowTable[focusBase + 2u];
    let maximumDistance = readParameter(5u);
    let maximumTimeLag = readParameter(6u);
    let spatialBandwidth = readParameter(9u);
    let temporalBandwidth = readParameter(10u);
    let focusUsable = isFiniteFloat(maximumDistance) && maximumDistance > 0.0 &&
      isFiniteFloat(maximumTimeLag) && maximumTimeLag > 0.0;
    let left = x - lattice.minimumX;
    let bottom = y - lattice.minimumY;
    let right = lattice.maximumX - x;
    let top = lattice.maximumY - y;
    var localBins: array<f32, ${hasLocal ? bins : 1}>;`,
    pairAction: `if (!focusUsable) { continue; }
            let otherBase = rowTableOffset + neighbor * 4u;
            let timeLag = abs(rowTable[otherBase] - focusTime);
            let spatialWeight = select(1.0, getSpaceTimeSpatialWeight(left, bottom, right, top, pairDistance), EDGE_ISOTROPIC);
            let temporalWeight = select(1.0, getSpaceTimeTemporalWeight(timeLag, focusRadius), EDGE_ISOTROPIC);
            let pairIntensity = focusInverse * rowTable[otherBase + 1u];
            let scaledDistance = pairDistance / maximumDistance * f32(DISTANCE_CLASS_COUNT);
            let scaledLag = timeLag / maximumTimeLag * f32(TIME_CLASS_COUNT);
            let distanceClass = min(u32(max(ceil(scaledDistance) - 1.0, 0.0)), DISTANCE_CLASS_COUNT - 1u);
            let lagClass = min(u32(max(ceil(scaledLag) - 1.0, 0.0)), TIME_CLASS_COUNT - 1u);
            let inDistance = scaledDistance <= f32(DISTANCE_CLASS_COUNT);
            let inLag = scaledLag <= f32(TIME_CLASS_COUNT);
            if (HAS_MARGINAL && inDistance) {
              let marginalSlot = BINS + distanceClass;
              accumulate(marginalSlot, 0u, 1u);
              accumulateScaled(marginalSlot, 1u, 2u, spatialWeight * WEIGHT_SCALE);
            }
            if (inDistance && inLag) {
              let slot = distanceClass * TIME_CLASS_COUNT + lagClass;
              let weight = spatialWeight * temporalWeight * pairIntensity;
              accumulate(slot, 0u, 1u);
              accumulateScaled(slot, 1u, 2u, weight * WEIGHT_SCALE);
              if (HAS_LOCAL) {
                localBins[slot] += weight;
              }
            }
            if (HAS_PAIR_CORRELATION && spatialBandwidth > 0.0 && temporalBandwidth > 0.0 &&
                pairDistance <= maximumDistance + spatialBandwidth &&
                timeLag <= maximumTimeLag + temporalBandwidth) {
              // Grid distance a has radius maximumDistance * (a + 1) / count.
              let firstDistance = max(i32(ceil((pairDistance - spatialBandwidth) / maximumDistance * f32(DISTANCE_CLASS_COUNT))) - 1, 0);
              let lastDistance = min(i32(floor((pairDistance + spatialBandwidth) / maximumDistance * f32(DISTANCE_CLASS_COUNT))) - 1, i32(DISTANCE_CLASS_COUNT) - 1);
              let firstLag = max(i32(ceil((timeLag - temporalBandwidth) / maximumTimeLag * f32(TIME_CLASS_COUNT))) - 1, 0);
              let lastLag = min(i32(floor((timeLag + temporalBandwidth) / maximumTimeLag * f32(TIME_CLASS_COUNT))) - 1, i32(TIME_CLASS_COUNT) - 1);
              let pairWeight = spatialWeight * temporalWeight * pairIntensity;
              for (var distanceIndex = firstDistance; distanceIndex <= lastDistance; distanceIndex++) {
                let spatialTerm = getSpatialPairKernel(maximumDistance * f32(distanceIndex + 1) / f32(DISTANCE_CLASS_COUNT) - pairDistance, spatialBandwidth);
                for (var lagIndex = firstLag; lagIndex <= lastLag; lagIndex++) {
                  let temporalTerm = getTemporalPairKernel(maximumTimeLag * f32(lagIndex + 1) / f32(TIME_CLASS_COUNT) - timeLag, temporalBandwidth);
                  accumulateScaled(u32(distanceIndex) * TIME_CLASS_COUNT + u32(lagIndex), 3u, 4u, pairWeight * spatialTerm * temporalTerm * WEIGHT_SCALE);
                }
              }
            }`,
    focusEpilogue: !hasLocal
      ? undefined
      : `if (HAS_LOCAL) {
      // 2D inclusive prefix of the local sums, in place.
      for (var distanceIndex = 0u; distanceIndex < DISTANCE_CLASS_COUNT; distanceIndex++) {
        for (var lagIndex = 0u; lagIndex < TIME_CLASS_COUNT; lagIndex++) {
          var value = localBins[distanceIndex * TIME_CLASS_COUNT + lagIndex];
          if (distanceIndex > 0u) { value += localBins[(distanceIndex - 1u) * TIME_CLASS_COUNT + lagIndex]; }
          if (lagIndex > 0u) { value += localBins[distanceIndex * TIME_CLASS_COUNT + lagIndex - 1u]; }
          if (distanceIndex > 0u && lagIndex > 0u) { value -= localBins[(distanceIndex - 1u) * TIME_CLASS_COUNT + lagIndex - 1u]; }
          localBins[distanceIndex * TIME_CLASS_COUNT + lagIndex] = value;
        }
      }
      // K_i = n * s * (sum over j), the scale s of the global estimate, so mean_i K_i = K.
      let includedCount = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
      let area = (lattice.maximumX - lattice.minimumX) * (lattice.maximumY - lattice.minimumY);
      let duration = readParameter(8u) - readParameter(7u);
      let scale = area * duration / select(includedCount * (includedCount - 1.0), includedCount * includedCount, INHOMOGENEOUS);
      for (var bin = 0u; bin < BINS; bin++) {
        local[localOffset + focus * BINS + bin] = includedCount * scale * localBins[bin];
      }
    }`
  });
  nodes.push(...histogram.nodes);

  // Temporal marginal on the sorted times, for the included rows.
  let temporalAccumulators: GraphDataView<'uint32'> | undefined;
  if (hasTemporal) {
    temporalAccumulators = createTransientView(
      graph,
      `${id}-temporal-accumulators`,
      'uint32',
      2 * V
    );
    const timeKeys = createTransientView(graph, `${id}-time-keys`, 'uint32', rows);
    const timeRowIds = createTransientView(graph, `${id}-time-row-ids`, 'uint32', rows);
    const sortedTimeKeys = createTransientView(graph, `${id}-sorted-time-keys`, 'uint32', rows);
    const sortedTimeRows = createTransientView(graph, `${id}-sorted-time-rows`, 'uint32', rows);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-time-keys`,
        operation,
        variant: 'time-keys',
        bindings: [
          {name: 'times', view: times, type: 'f32', access: 'read'},
          {name: 'cellKeys', view: inputs.cellKeys, type: 'u32', access: 'read'},
          {name: 'timeKeys', view: timeKeys, type: 'u32', access: 'read_write'},
          {name: 'timeRowIds', view: timeRowIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${GET_ORDERED_KEY_WGSL}
const CELL_COUNT: u32 = ${cellCount}u;`,
        body: `// Rows the pair index excluded sort last.
  let included = cellKeys[cellKeysOffset + index] < CELL_COUNT;
  timeKeys[timeKeysOffset + index] = select(0xffffffffu, getOrderedKey(times[timesOffset + index]), included);
  timeRowIds[timeRowIdsOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-time-sort`,
        keys: timeKeys,
        values: timeRowIds,
        outputKeys: sortedTimeKeys,
        outputValues: sortedTimeRows,
        keyBits: 32
      }).getCommandNodes(graph),
      createFillNode<Parameters>(graph, {
        id: `${id}-temporal-clear`,
        operation,
        view: temporalAccumulators,
        type: 'u32',
        value: '0u'
      }),
      createTemporalMarginalNode(graph, {
        id,
        operation,
        sortedKeys: sortedTimeKeys,
        cellOffsets: inputs.cellOffsets,
        parameters,
        accumulators: temporalAccumulators,
        timeClassCount: V,
        cellCount,
        edgeCorrection: config.edgeCorrection
      })
    );
  }

  // Finish: cumulative K and the marginals, pair counts, then D, the pair correlation and status.
  const readAccumulatorsWGSL = getPairHistogramReadWGSL(
    'accumulators',
    channelCount,
    '',
    slotCount
  );
  const finishBindings: WGSLKernelBinding[] = [
    {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
    {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
    {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
    {name: 'k', view: config.k, type: 'f32', access: 'read_write'},
    ...(config.spatialK
      ? [{name: 'spatialK', view: config.spatialK, type: 'f32', access: 'read_write'} as const]
      : []),
    ...(temporalAccumulators && config.temporalK
      ? [
          {
            name: 'temporalAccumulators',
            view: temporalAccumulators,
            type: 'u32',
            access: 'read'
          } as const,
          {name: 'temporalK', view: config.temporalK, type: 'f32', access: 'read_write'} as const
        ]
      : []),
    ...(config.pairCounts
      ? [{name: 'pairCounts', view: config.pairCounts, type: 'u32', access: 'read_write'} as const]
      : [])
  ];
  const scaleWGSL = `let includedCount = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let area = (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]);
  let duration = parameters[parametersOffset + 8u] - parameters[parametersOffset + 7u];
  let defined = includedCount >= 2.0 && isFiniteFloat(area) && area > 0.0 &&
    isFiniteFloat(duration) && duration > 0.0 &&
    parameters[parametersOffset + 5u] > 0.0 && parameters[parametersOffset + 6u] > 0.0;
  let pairNormalizer = select(includedCount * (includedCount - 1.0), includedCount * includedCount, INHOMOGENEOUS);`;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-finish`,
      operation,
      variant: `finish-${config.edgeCorrection}-${Boolean(config.intensities)}-${hasMarginal}-${hasTemporal}-${Boolean(config.pairCounts)}-${hasPairCorrelation}`,
      bindings: finishBindings,
      invocationCount: bins,
      declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${constants}
const CELL_COUNT: u32 = ${cellCount}u;
const HAS_MARGINAL: bool = ${hasMarginal};
${readAccumulatorsWGSL}`,
      body: `${scaleWGSL}
  let distanceIndex = index / TIME_CLASS_COUNT;
  let lagIndex = index % TIME_CLASS_COUNT;
  var sum = 0.0;
  for (var firstIndex = 0u; firstIndex <= distanceIndex; firstIndex++) {
    for (var secondIndex = 0u; secondIndex <= lagIndex; secondIndex++) {
      let slot = firstIndex * TIME_CLASS_COUNT + secondIndex;
      sum += readPairAccumulator(slot, 1u) + readPairAccumulator(slot, 2u) * LARGE_UNIT;
    }
  }
  k[kOffset + index] = select(getQuietNaN(index), area * duration / pairNormalizer * sum / WEIGHT_SCALE, defined);
  ${
    config.spatialK
      ? `if (index < DISTANCE_CLASS_COUNT) {
    var spatialSum = 0.0;
    for (var firstIndex = 0u; firstIndex <= index; firstIndex++) {
      spatialSum += readPairAccumulator(BINS + firstIndex, 1u) + readPairAccumulator(BINS + firstIndex, 2u) * LARGE_UNIT;
    }
    spatialK[spatialKOffset + index] = select(getQuietNaN(index), area / pairNormalizer * spatialSum / WEIGHT_SCALE, defined);
  }`
      : ''
  }
  ${
    temporalAccumulators && config.temporalK
      ? `if (index < TIME_CLASS_COUNT) {
    let word = index * 2u;
    let temporalSum = f32(temporalAccumulators[temporalAccumulatorsOffset + word + 1u]) * 4294967296.0 + f32(temporalAccumulators[temporalAccumulatorsOffset + word]);
    temporalK[temporalKOffset + index] = select(getQuietNaN(index), duration / pairNormalizer * temporalSum, defined);
  }`
      : ''
  }
  ${config.pairCounts ? 'pairCounts[pairCountsOffset + index] = readPairCount(index, 0u);' : ''}`
    })
  );

  if (config.d && config.spatialK && config.temporalK) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-d`,
        operation,
        variant: 'diggle-d',
        bindings: [
          {name: 'k', view: config.k, type: 'f32', access: 'read'},
          {name: 'spatialK', view: config.spatialK, type: 'f32', access: 'read'},
          {name: 'temporalK', view: config.temporalK, type: 'f32', access: 'read'},
          {name: 'd', view: config.d, type: 'f32', access: 'read_write'}
        ],
        invocationCount: bins,
        body: `d[dOffset + index] = k[kOffset + index] -
    spatialK[spatialKOffset + index / ${V}u] * temporalK[temporalKOffset + index % ${V}u];`
      })
    );
  }

  if (config.pairCorrelation) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-correlation`,
        operation,
        variant: 'pair-correlation',
        bindings: [
          {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
          {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'pairCorrelation', view: config.pairCorrelation, type: 'f32', access: 'read_write'}
        ],
        invocationCount: bins,
        declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${constants}
const CELL_COUNT: u32 = ${cellCount}u;
${readAccumulatorsWGSL}`,
        body: `${scaleWGSL}
  let spatialBandwidth = parameters[parametersOffset + 9u];
  let temporalBandwidth = parameters[parametersOffset + 10u];
  let distanceIndex = index / TIME_CLASS_COUNT;
  let radius = parameters[parametersOffset + 5u] * f32(distanceIndex + 1u) / f32(DISTANCE_CLASS_COUNT);
  let sum = readPairAccumulator(index, 3u) + readPairAccumulator(index, 4u) * LARGE_UNIT;
  // g = s / (4 pi u) * sum ks kt w / (lambda lambda'), with the kernels' bandwidths undone.
  let value = area * duration / pairNormalizer * sum / WEIGHT_SCALE / (4.0 * 3.14159265358979 * radius * spatialBandwidth * temporalBandwidth);
  pairCorrelation[pairCorrelationOffset + index] = select(getQuietNaN(index), value, defined && spatialBandwidth > 0.0 && temporalBandwidth > 0.0);`
      })
    );
  }

  if (config.status) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-status`,
        operation,
        variant: 'status',
        bindings: [
          {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
          {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'status', view: config.status, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${constants}
const CELL_COUNT: u32 = ${cellCount}u;
${readAccumulatorsWGSL}`,
        body: `${scaleWGSL}
  // 0 valid, 1 undefined (fewer than two included rows or an invalid window or maximum),
  // 3 the exact integer accumulation overflowed.
  status[statusOffset] = select(select(0u, 3u, readPairOverflow()), 1u, !defined);`
      })
    );
  }
  return {nodes, inputs, rowTable, effectiveMask};
}

/**
 * The temporal marginal `sum_i sum_{j != i} 1(|t_i - t_j| <= v_b) w^t_ij`, exact integers, for the
 * included rows sorted by time. For each sorted event and class it counts the partners inside the
 * lag by binary search on the sorted keys, with the class test of the pair pass (`lag / v * count
 * <= b + 1`), so the two paths agree on every boundary. The isotropic weight is 2 for partners
 * beyond the event's edge radius and 1 inside, which gives `2 N(v) - N(min(v, radius))`.
 */
function createTemporalMarginalNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    sortedKeys: GraphDataView<'uint32'>;
    cellOffsets: GraphDataView<'uint32'>;
    parameters: GraphDataView<'float32'>;
    accumulators: GraphDataView<'uint32'>;
    timeClassCount: number;
    cellCount: number;
    edgeCorrection: GPUSpaceTimeKEdgeCorrection;
  }
): GPUCommandNode<Parameters> {
  const workgroupSize = 64;
  const V = props.timeClassCount;
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-temporal-marginal`,
    operation: props.operation,
    variant: `temporal-marginal-${props.edgeCorrection}`,
    bindings: [
      {name: 'sortedKeys', view: props.sortedKeys, type: 'u32', access: 'read'},
      {name: 'cellOffsets', view: props.cellOffsets, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'accumulators', view: props.accumulators, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.sortedKeys.length,
    workgroupSize,
    guardIndex: false,
    declarations: `${DECODE_ORDERED_KEY_WGSL}
const CELL_COUNT: u32 = ${props.cellCount}u;
const TIME_CLASS_COUNT: u32 = ${V}u;
const EDGE_ISOTROPIC: bool = ${props.edgeCorrection === 'isotropic'};
// One workgroup holds 64 rows, each adding at most 2 n < 2^25 to a class, so a u32 never wraps.
var<workgroup> localCounts: array<atomic<u32>, ${V}>;

fn isInside(lag: f32, limit: f32, radius: f32, useRadius: bool, maximumTimeLag: f32) -> bool {
  return lag / maximumTimeLag * f32(TIME_CLASS_COUNT) <= limit && (!useRadius || lag <= radius);
}

// Number of sorted partners j != i inside the lag, for the event at the given time whose first
// equal key is at index first.
fn countPartners(time: f32, first: u32, count: u32, limit: f32, radius: f32, useRadius: bool, maximumTimeLag: f32) -> u32 {
  // Partners at or after the event: the first index failing the test.
  var low = first;
  var high = count;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (isInside(decodeOrderedKey(sortedKeys[sortedKeysOffset + middle]) - time, limit, radius, useRadius, maximumTimeLag)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let after = low - first;
  // Partners before the event: the first index passing the test.
  low = 0u;
  high = first;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (isInside(time - decodeOrderedKey(sortedKeys[sortedKeysOffset + middle]), limit, radius, useRadius, maximumTimeLag)) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  // The event itself is among the partners at or after it.
  return after + (first - low) - 1u;
}`,
    body: `for (var word = localInvocationIndex; word < TIME_CLASS_COUNT; word += ${workgroupSize}u) {
    atomicStore(&localCounts[word], 0u);
  }
  workgroupBarrier();
  let includedCount = cellOffsets[cellOffsetsOffset + CELL_COUNT];
  let maximumTimeLag = parameters[parametersOffset + 6u];
  if (index < includedCount && maximumTimeLag > 0.0) {
    let key = sortedKeys[sortedKeysOffset + index];
    let time = decodeOrderedKey(key);
    let radius = min(time - parameters[parametersOffset + 7u], parameters[parametersOffset + 8u] - time);
    // First index holding this key.
    var first = 0u;
    var end = index;
    while (first < end) {
      let middle = first + (end - first) / 2u;
      if (sortedKeys[sortedKeysOffset + middle] < key) {
        first = middle + 1u;
      } else {
        end = middle;
      }
    }
    for (var timeClass = 0u; timeClass < TIME_CLASS_COUNT; timeClass++) {
      let limit = f32(timeClass + 1u);
      var weighted = countPartners(time, first, includedCount, limit, radius, false, maximumTimeLag);
      if (EDGE_ISOTROPIC) {
        weighted = 2u * weighted - countPartners(time, first, includedCount, limit, radius, true, maximumTimeLag);
      }
      atomicAdd(&localCounts[timeClass], weighted);
    }
  }
  workgroupBarrier();
  for (var word = localInvocationIndex; word < TIME_CLASS_COUNT; word += ${workgroupSize}u) {
    let low = atomicLoad(&localCounts[word]);
    if (low != 0u) {
      // Cumulative in the class index: the per-class counts already are, so add them as they are.
      let previous = atomicAdd(&accumulators[accumulatorsOffset + word * 2u], low);
      if (previous > 0xffffffffu - low) {
        atomicAdd(&accumulators[accumulatorsOffset + word * 2u + 1u], 1u);
      }
    }
  }`
  });
}
