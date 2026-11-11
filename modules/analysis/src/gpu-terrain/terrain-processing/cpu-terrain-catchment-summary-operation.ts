// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getTerrainDrainageDirection,
  getTerrainDrainageReceiver,
  TERRAIN_DRAINAGE_NONE,
  TERRAIN_DRAINAGE_UNRESOLVED
} from '../hydrology/cpu-terrain-drainage-utils';
import {validateTerrainFlowDirectionEncoding} from '../hydrology/terrain-flow-tracing';
import {
  getCPUTerrainGroundMeasures,
  TERRAIN_OPERATION_MAXIMUM_WORD_COUNT,
  validateCPUTerrainOperationGrid
} from './cpu-terrain-operation-grid';
import {
  GPUTerrainCatchmentSummaryOperation,
  type TerrainCatchmentSummaryOperationProps
} from './gpu-terrain-catchment-summary-operation';

export type CPUTerrainCatchmentSummaryInput = {
  elevation: Float32Array;
  validity?: Uint32Array;
  basinIds: Uint32Array;
  streams: Uint32Array;
  flowDirections: Uint32Array;
};
export type CPUTerrainCatchmentSummaryOptions = Pick<
  TerrainCatchmentSummaryOperationProps,
  'grid' | 'cellSizeMode' | 'directionEncoding' | 'basinCount' | 'histogramBins' | 'maximumSteps'
>;
export type CPUTerrainCatchmentSummaryResult = {
  statistics: Float32Array;
  hypsometry: Float32Array;
  profiles: Float32Array;
  profileStatus: Uint32Array;
};

/**
 * Aggregates basin statistics and cumulative upper-edge histograms, then memoizes stream paths.
 * Each edge belongs to its source basin, even across a basin boundary. Profiles traverse through
 * excluded basin IDs; only the starting cell's basin determines whether its output is excluded.
 * maximumSteps is retained: the number of stream edges must be strictly less than it.
 *
 * What differs from the GPU contributor:
 * - Float64 ground measures, compensated elevation sums, means, thresholds and profiles round
 *   once at output, instead of fixed-point sums of float32 terms and float32 pointer-jumping.
 *   Compare nonzero finite arithmetic results with relative tolerance 2e-5. Copied elevations
 *   and extrema agree exactly; integer statuses agree away from threshold/overflow boundaries.
 *   Values close to a hypsometry edge can enter a different bin.
 * - Histograms accumulate each cell once and prefix-sum bins; profile walks have no GPU round
 *   budget. The documented exclusive maximumSteps bound and cycle/failure statuses are kept.
 * - A final distance outside float32 becomes NaN/status 2; finite float64 intermediates can
 *   resolve a path that overflows on GPU. Subnormals are retained rather than flushed to zero.
 *
 * @throws If any input array does not contain exactly one value per grid cell.
 */
export function computeTerrainCatchmentSummaryOnCPU(
  input: CPUTerrainCatchmentSummaryInput,
  options: CPUTerrainCatchmentSummaryOptions
): CPUTerrainCatchmentSummaryResult {
  const count = validateSummaryOptions('computeTerrainCatchmentSummaryOnCPU', options);
  for (const [name, array] of Object.entries(input))
    if (array !== undefined && array.length !== count)
      throw new Error(`${name} must contain one value per grid cell`);
  const {grid, basinCount, histogramBins} = options;
  const elevation = Float64Array.from(input.elevation, (value, cell) =>
    (input.validity?.[cell] ?? 1) !== 0 ? value : NaN
  );
  const measures = getCPUTerrainGroundMeasures(grid, options.cellSizeMode);
  const isStream = (cell: number) =>
    input.streams[cell] !== 0 && input.streams[cell] < TERRAIN_DRAINAGE_UNRESOLVED;
  const receivers = Uint32Array.from(input.flowDirections, (word, cell) =>
    getTerrainDrainageReceiver(
      cell,
      getTerrainDrainageDirection(word, options.directionEncoding),
      grid.width,
      grid.height
    )
  );
  const sampleCounts = new Uint32Array(basinCount);
  const lower = new Float64Array(basinCount).fill(Infinity);
  const upper = new Float64Array(basinCount).fill(-Infinity);
  const sums = new Float64Array(basinCount);
  const sumCorrections = new Float64Array(basinCount);
  const areas = new Float64Array(basinCount);
  const streamLengths = new Float64Array(basinCount);
  const bins = new Float64Array(basinCount * histogramBins);
  for (let cell = 0; cell < count; cell++) {
    const basin = input.basinIds[cell],
      value = elevation[cell];
    if (basin >= basinCount || !Number.isFinite(value)) continue;
    sampleCounts[basin]++;
    lower[basin] = Math.min(lower[basin], value);
    upper[basin] = Math.max(upper[basin], value);
    // Neumaier compensation preserves small elevations when large opposite signs cancel.
    // These float64 error terms execute on the CPU, outside shader fast-math reassociation.
    const previousSum = sums[basin];
    const nextSum = previousSum + value;
    sumCorrections[basin] +=
      Math.abs(previousSum) >= Math.abs(value)
        ? previousSum - nextSum + value
        : value - nextSum + previousSum;
    sums[basin] = nextSum;
    areas[basin] += measures.cellArea(cell);
    const receiver = receivers[cell];
    if (
      isStream(cell) &&
      receiver !== TERRAIN_DRAINAGE_NONE &&
      isStream(receiver) &&
      Number.isFinite(elevation[receiver])
    )
      streamLengths[basin] += measures.edgeLength(cell, receiver);
  }
  for (let cell = 0; cell < count; cell++) {
    const basin = input.basinIds[cell],
      value = elevation[cell];
    if (basin >= basinCount || !Number.isFinite(value)) continue;
    // Find the first inclusive upper edge. Binary search avoids cancellation in ceil(normalized).
    let first = 0,
      last = histogramBins - 1;
    while (first < last) {
      const middle = Math.floor((first + last) / 2);
      const threshold =
        lower[basin] + ((upper[basin] - lower[basin]) * (middle + 1)) / histogramBins;
      if (value <= threshold) last = middle;
      else first = middle + 1;
    }
    bins[basin * histogramBins + first] += measures.cellArea(cell);
  }
  const statistics = new Float32Array(basinCount * 6);
  const hypsometry = new Float32Array(basinCount * histogramBins);
  for (let basin = 0; basin < basinCount; basin++) {
    statistics.set(
      [
        areas[basin],
        sampleCounts[basin] ? (sums[basin] + sumCorrections[basin]) / sampleCounts[basin] : NaN,
        sampleCounts[basin] ? lower[basin] : NaN,
        sampleCounts[basin] ? upper[basin] : NaN,
        streamLengths[basin],
        areas[basin] > 0 ? streamLengths[basin] / Math.max(areas[basin], 1e-30) : NaN
      ],
      basin * 6
    );
    let cumulative = 0;
    for (let bin = 0; bin < histogramBins; bin++) {
      cumulative += bins[basin * histogramBins + bin];
      hypsometry[basin * histogramBins + bin] =
        bin + 1 === histogramBins ? areas[basin] : cumulative;
    }
  }
  const pathReceivers = new Uint32Array(count).fill(TERRAIN_DRAINAGE_NONE);
  const distances = new Float64Array(count);
  const steps = new Float64Array(count);
  const complete = new Uint8Array(count);
  const state = new Uint8Array(count);
  for (let cell = 0; cell < count; cell++) {
    state[cell] = 2;
    if (!Number.isFinite(elevation[cell]) || !isStream(cell)) continue;
    const code = getTerrainDrainageDirection(input.flowDirections[cell], options.directionEncoding);
    if (code === 0) {
      complete[cell] = 1;
      continue;
    }
    const receiver = receivers[cell];
    if (
      receiver === TERRAIN_DRAINAGE_NONE ||
      !Number.isFinite(elevation[receiver]) ||
      input.streams[receiver] >= TERRAIN_DRAINAGE_UNRESOLVED
    )
      continue;
    if (input.streams[receiver] === 0) {
      complete[cell] = 1;
      continue;
    }
    state[cell] = 0;
    pathReceivers[cell] = receiver;
    distances[cell] = measures.edgeLength(cell, receiver);
  }
  for (let start = 0; start < count; start++) {
    if (state[start] === 2) continue;
    let cell = start;
    const path: number[] = [];
    while (state[cell] === 0) {
      state[cell] = 1;
      path.push(cell);
      cell = pathReceivers[cell];
    }
    const resolved = state[cell] === 2 && complete[cell] !== 0;
    let distance = resolved ? distances[cell] : 0;
    let length = resolved ? steps[cell] : 0;
    for (let position = path.length - 1; position >= 0; position--) {
      const member = path[position];
      distance += distances[member];
      length++;
      distances[member] = distance;
      steps[member] = length;
      complete[member] = Number(resolved);
      state[member] = 2;
    }
  }
  const profiles = new Float32Array(count * 3).fill(NaN);
  const profileStatus = new Uint32Array(count);
  for (let cell = 0; cell < count; cell++) {
    profiles[cell * 3 + 1] = elevation[cell];
    if (input.streams[cell] >= TERRAIN_DRAINAGE_UNRESOLVED) {
      profileStatus[cell] = 2;
      continue;
    }
    if (!Number.isFinite(elevation[cell]) || input.basinIds[cell] >= basinCount || !isStream(cell))
      continue;
    profileStatus[cell] = 2;
    if (
      !complete[cell] ||
      steps[cell] >= (options.maximumSteps ?? count) ||
      !Number.isFinite(Math.fround(distances[cell]))
    )
      continue;
    const receiver = receivers[cell];
    const gradient =
      receiver !== TERRAIN_DRAINAGE_NONE && isStream(receiver)
        ? (elevation[cell] - elevation[receiver]) / measures.edgeLength(cell, receiver)
        : 0;
    profileStatus[cell] = 1;
    profiles[cell * 3] = distances[cell];
    profiles[cell * 3 + 2] = gradient;
  }
  return {statistics, hypsometry, profiles, profileStatus};
}

function validateSummaryOptions(id: string, options: CPUTerrainCatchmentSummaryOptions): number {
  const count = validateCPUTerrainOperationGrid(id, options.grid, options.cellSizeMode);
  validateTerrainFlowDirectionEncoding(id, options.directionEncoding);
  validateCapacity(id, 'basinCount', options.basinCount);
  validateCapacity(id, 'histogramBins', options.histogramBins);
  validateCapacity(id, 'maximumSteps', options.maximumSteps ?? count);
  validateMaximumCount(
    id,
    'basinCount * histogramBins',
    options.basinCount * options.histogramBins,
    TERRAIN_OPERATION_MAXIMUM_WORD_COUNT
  );
  validateMaximumCount(
    id,
    'basinCount * 6',
    options.basinCount * 6,
    TERRAIN_OPERATION_MAXIMUM_WORD_COUNT
  );
  validateRequirement(
    id,
    Math.abs(
      options.grid.affine[0] * options.grid.affine[4] -
        options.grid.affine[1] * options.grid.affine[3]
    ) > 0,
    'grid.affine must have a nonzero determinant'
  );
  return count;
}

/** CPU summary twin; all output layouts and the exclusive maximumSteps definition are retained. */
export class CPUTerrainCatchmentSummaryOperation extends GPUCPUOperation {
  constructor(props: TerrainCatchmentSummaryOperationProps) {
    const id = props.id ?? 'terrain-catchment-summary';
    validateSummaryOptions(id, props);
    const operation = new GPUTerrainCatchmentSummaryOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const result = computeTerrainCatchmentSummaryOnCPU(
          {
            elevation: getCPUOperationFloat32Rows(id, inputs, 'elevation'),
            basinIds: getCPUOperationUint32Rows(id, inputs, 'basinIds'),
            streams: getCPUOperationUint32Rows(id, inputs, 'streams'),
            flowDirections: getCPUOperationUint32Rows(id, inputs, 'flowDirections'),
            validity:
              props.validity !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'validity')
                : undefined
          },
          props
        );
        return {
          [props.output.statistics]: result.statistics,
          [props.output.hypsometry]: result.hypsometry,
          [props.output.profiles]: result.profiles,
          [props.output.profileStatus]: result.profileStatus
        };
      }
    });
  }
}
