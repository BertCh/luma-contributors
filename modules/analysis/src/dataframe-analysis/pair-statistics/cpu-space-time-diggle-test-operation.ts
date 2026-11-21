// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {fillShufflePermutation} from '../../analysis-core/kernels/permutation-shuffle';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH} from './space-time-k-function-nodes';
import {GPU_SPACE_TIME_K_WEIGHT_CAP} from './space-time-k-function-wgsl';
import {
  GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH,
  GPU_SPACE_TIME_DIGGLE_SUMMARY,
  GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH
} from './gpu-space-time-diggle-test';
import {
  GPUSpaceTimeDiggleTestOperation,
  snapshotSpaceTimeDiggleOperationProps,
  type SpaceTimeDiggleTestOperationProps
} from './gpu-space-time-diggle-test-operation';
import {
  getRectangleIsotropicWeight,
  getRectangleSides,
  isRectangleLatticeValid
} from './cpu-inference-rectangle';

/** Packed graph inputs; masks select rows before the shuffle domain is formed. */
export type CPUSpaceTimeDiggleTestInput = {
  positions: Float32Array;
  times: Float32Array;
  parameters: Float32Array;
  randomParameters: Uint32Array;
  mask?: Uint32Array;
};
/** All histogram and permutation limits specific to the GPU are dropped. */
export type CPUSpaceTimeDiggleTestOptions = Pick<
  SpaceTimeDiggleTestOperationProps,
  'distanceClassCount' | 'timeClassCount' | 'permutationCount' | 'edgeCorrection'
>;
/** All supported outputs; classes are distance-major. */
export type CPUSpaceTimeDiggleTestResult = {
  d: Float32Array;
  statistics: Float32Array;
  summary: Float32Array;
  permutationMean: Float32Array;
  permutationStandardDeviation: Float32Array;
  k: Float32Array;
  spatialK: Float32Array;
  temporalK: Float32Array;
  status: Uint32Array;
};

/** One statistics row is reserved for the observed value in a uint32-sized vector. */
export const CPU_SPACE_TIME_DIGGLE_MAXIMUM_PERMUTATIONS = 0xfffffffe;

/** Every class-grid output is a uint32-indexed vector. */
export const CPU_SPACE_TIME_DIGGLE_MAXIMUM_BIN_COUNT = 0xffffffff;

function validateOptions(id: string, options: CPUSpaceTimeDiggleTestOptions): void {
  validateCapacity(id, 'distanceClassCount', options.distanceClassCount);
  validateCapacity(id, 'timeClassCount', options.timeClassCount);
  validateCapacity(
    id,
    'permutationCount',
    options.permutationCount,
    1,
    CPU_SPACE_TIME_DIGGLE_MAXIMUM_PERMUTATIONS
  );
  validateMaximumCount(
    id,
    'distanceClassCount * timeClassCount',
    options.distanceClassCount * options.timeClassCount,
    CPU_SPACE_TIME_DIGGLE_MAXIMUM_BIN_COUNT
  );
  validateRequirement(
    id,
    options.edgeCorrection === undefined || ['none', 'isotropic'].includes(options.edgeCorrection),
    'edgeCorrection must be none or isotropic'
  );
}

/** First class whose product bound includes value; the final upper bound is open (splancs). */
function getClassIndex(value: number, maximum: number, count: number): number {
  if (!(value < maximum)) return count;
  let lower = 0;
  let upper = count;
  while (lower < upper) {
    const middle = lower + Math.floor((upper - lower) / 2);
    if (value * count <= maximum * (middle + 1)) upper = middle;
    else lower = middle + 1;
  }
  return lower;
}

/**
 * Diggle's sum of D=K-K_s K_t under random time relabelling, with the current splancs rules:
 * largest distance/lag excluded, interior class bounds included, temporal edge ties weighted 2.
 *
 * What differs from the GPU contributor:
 * - All calculations and inference comparisons use float64, rounding only final output arrays.
 *   Ordinary K/marginal/standard-deviation values allow 2e-4 relative error; signed residuals
 *   and sums allow 2e-4 of the contributing K scale. Near ties can change permutation counts.
 * - The registered Philox shuffle is unchanged, over included rows in original row order.
 * - GPU-only 64-class, 999-permutation, histogram, binding-size and command-node caps are
 *   dropped; uint32 output lengths remain. All ordered pairs are visited without a grid index.
 * - Isotropic weights retain the estimator's cap of 100; fixed-point quantization is omitted.
 *   Status 3 (GPU integer accumulator overflow) is never set; 0 and 1 keep their meanings.
 * - Nonfinite/out-of-window positions or times are excluded. Finite bounds, squared search
 *   radii and areas that overflow float32 remain usable when the float64 result is finite.
 *
 * @throws If positions/times/mask lengths disagree or either parameter layout is incomplete.
 */
export function computeSpaceTimeDiggleTestOnCPU(
  input: CPUSpaceTimeDiggleTestInput,
  options: CPUSpaceTimeDiggleTestOptions
): CPUSpaceTimeDiggleTestResult {
  validateOptions('space-time-diggle-test', options);
  const {positions, times, parameters, randomParameters, mask} = input;
  if (positions.length === 0 || positions.length % 2 !== 0)
    throw new Error('positions must contain nonempty float32x2 rows');
  const rows = positions.length / 2;
  if (times.length !== rows) throw new Error('times length must equal positions row count');
  if (mask && mask.length !== rows) throw new Error('mask length must equal positions row count');
  if (parameters.length < GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH)
    throw new Error('parameters must hold getGPUSpaceTimeKFunctionParameterValues layout');
  if (randomParameters.length < GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH)
    throw new Error('randomParameters must hold getGPUSpaceTimeDiggleRandomParameterValues layout');
  const [
    minimumX,
    minimumY,
    maximumX,
    maximumY,
    searchDistance,
    maximumDistance,
    maximumTimeLag,
    timeStart,
    timeEnd
  ] = parameters;
  const bounds = [minimumX, minimumY, maximumX, maximumY];
  const latticeValid = isRectangleLatticeValid(bounds, searchDistance);
  const included: number[] = [];
  for (let row = 0; row < rows; row++) {
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    const time = times[row];
    if (
      latticeValid &&
      (mask === undefined || mask[row] !== 0) &&
      Number.isFinite(timeStart) &&
      Number.isFinite(timeEnd) &&
      timeStart < timeEnd &&
      Number.isFinite(time) &&
      time >= timeStart &&
      time <= timeEnd &&
      Number.isFinite(x) &&
      Number.isFinite(y) &&
      x >= minimumX &&
      x <= maximumX &&
      y >= minimumY &&
      y <= maximumY
    )
      included.push(row);
  }
  const distanceCount = options.distanceClassCount;
  const timeCount = options.timeClassCount;
  const bins = distanceCount * timeCount;
  const result: CPUSpaceTimeDiggleTestResult = {
    d: new Float32Array(bins).fill(NaN),
    statistics: new Float32Array(options.permutationCount + 1).fill(NaN),
    summary: new Float32Array(GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH).fill(NaN),
    permutationMean: new Float32Array(bins).fill(NaN),
    permutationStandardDeviation: new Float32Array(bins).fill(NaN),
    k: new Float32Array(bins).fill(NaN),
    spatialK: new Float32Array(distanceCount).fill(NaN),
    temporalK: new Float32Array(timeCount).fill(NaN),
    status: Uint32Array.of(1)
  };
  result.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.usedClasses] = 0;
  result.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.includedCount] = included.length;
  const area = (maximumX - minimumX) * (maximumY - minimumY);
  const duration = timeEnd - timeStart;
  if (
    !(
      included.length >= 2 &&
      Number.isFinite(area) &&
      area > 0 &&
      Number.isFinite(duration) &&
      duration > 0 &&
      maximumDistance > 0 &&
      maximumTimeLag > 0
    )
  )
    return result;
  const normalizer = included.length * (included.length - 1);
  const isotropic = options.edgeCorrection !== 'none';
  const spatial = new Float64Array(distanceCount);
  const temporal = new Float64Array(timeCount);
  const pairs: {focus: number; other: number; distanceClass: number; spatialWeight: number}[] = [];
  const usableMaximums = Number.isFinite(maximumDistance) && Number.isFinite(maximumTimeLag);
  for (let focus = 0; focus < included.length; focus++) {
    const row = included[focus];
    const sides = getRectangleSides(positions[row * 2], positions[row * 2 + 1], bounds);
    for (let other = 0; other < included.length; other++)
      if (focus !== other) {
        const neighbor = included[other];
        const distance = Math.hypot(
          positions[row * 2] - positions[neighbor * 2],
          positions[row * 2 + 1] - positions[neighbor * 2 + 1]
        );
        const distanceClass = getClassIndex(distance, maximumDistance, distanceCount);
        if (usableMaximums && distance <= searchDistance && distanceClass < distanceCount) {
          const spatialWeight = isotropic
            ? getRectangleIsotropicWeight(sides, distance, GPU_SPACE_TIME_K_WEIGHT_CAP)
            : 1;
          pairs.push({focus, other, distanceClass, spatialWeight});
          for (let index = distanceClass; index < distanceCount; index++)
            spatial[index] += spatialWeight;
        }
        const lag = Math.abs(times[row] - times[neighbor]);
        const lagClass = getClassIndex(lag, maximumTimeLag, timeCount);
        const temporalWeight =
          isotropic && Math.min(times[row] - timeStart, timeEnd - times[row]) <= lag ? 2 : 1;
        for (let index = lagClass; index < timeCount; index++) temporal[index] += temporalWeight;
      }
  }
  for (let index = 0; index < distanceCount; index++) spatial[index] *= area / normalizer;
  for (let index = 0; index < timeCount; index++) temporal[index] *= duration / normalizer;
  const products = Float64Array.from(
    {length: bins},
    (_, index) => spatial[Math.floor(index / timeCount)] * temporal[index % timeCount]
  );
  const preciseStatistics = new Float64Array(options.permutationCount + 1);
  const classMeans = new Float64Array(bins);
  const classSquares = new Float64Array(bins);
  const permutation = new Uint32Array(included.length);
  const [seedLow, seedHigh] = randomParameters;
  for (let slot = 0; slot <= options.permutationCount; slot++) {
    if (slot > 0)
      fillShufflePermutation(
        permutation,
        included.length,
        [seedLow, seedHigh],
        slot,
        PHILOX_PURPOSES.spaceTimeDiggleShuffle.id
      );
    const permutedTimes = included.map(
      (row, rank) => times[slot === 0 ? row : included[permutation[rank]]]
    );
    const curve = new Float64Array(bins);
    for (const pair of pairs) {
      const focusTime = permutedTimes[pair.focus];
      const lag = Math.abs(focusTime - permutedTimes[pair.other]);
      const lagClass = getClassIndex(lag, maximumTimeLag, timeCount);
      if (lagClass >= timeCount) continue;
      const temporalWeight =
        isotropic && Math.min(focusTime - timeStart, timeEnd - focusTime) <= lag ? 2 : 1;
      // Suffix updates give the cumulative class grid directly, without a quantized histogram.
      for (let distance = pair.distanceClass; distance < distanceCount; distance++)
        for (let time = lagClass; time < timeCount; time++)
          curve[distance * timeCount + time] += pair.spatialWeight * temporalWeight;
    }
    for (let bin = 0; bin < bins; bin++) {
      const kValue = ((area * duration) / normalizer) * curve[bin];
      const residual = kValue - products[bin];
      preciseStatistics[slot] += residual;
      if (slot === 0) {
        result.k[bin] = kValue;
        result.d[bin] = residual;
      } else {
        const difference = residual - classMeans[bin];
        classMeans[bin] += difference / slot;
        classSquares[bin] += difference * (residual - classMeans[bin]);
      }
    }
  }
  result.spatialK.set(spatial);
  result.temporalK.set(temporal);
  result.statistics.set(preciseStatistics);
  result.permutationMean.set(classMeans);
  if (options.permutationCount > 1)
    result.permutationStandardDeviation.set(
      Float64Array.from(classSquares, value =>
        Math.sqrt(Math.max(0, value) / (options.permutationCount - 1))
      )
    );
  const observed = preciseStatistics[0];
  let mean = 0;
  let greater = 0;
  let lesser = 0;
  for (let slot = 1; slot <= options.permutationCount; slot++) {
    mean += preciseStatistics[slot] / options.permutationCount;
    if (preciseStatistics[slot] >= observed) greater++;
    if (preciseStatistics[slot] <= observed) lesser++;
  }
  let squares = 0;
  for (let slot = 1; slot <= options.permutationCount; slot++)
    squares += (preciseStatistics[slot] - mean) ** 2;
  const variance = options.permutationCount > 1 ? squares / (options.permutationCount - 1) : 0;
  const greaterProbability = (greater + 1) / (options.permutationCount + 1);
  const lesserProbability = (lesser + 1) / (options.permutationCount + 1);
  result.summary.set([
    observed,
    mean,
    variance,
    greater,
    lesser,
    greaterProbability,
    lesserProbability,
    Math.min(1, 2 * Math.min(greaterProbability, lesserProbability)),
    variance > 0 ? (observed - mean) / Math.sqrt(variance) : NaN,
    bins,
    included.length
  ]);
  result.status[0] = 0;
  return result;
}

/** CPU twin, without GPU class/histogram/permutation or node-budget limits. */
export class CPUSpaceTimeDiggleTestOperation extends GPUCPUOperation {
  constructor(inputProps: SpaceTimeDiggleTestOperationProps) {
    const props = snapshotSpaceTimeDiggleOperationProps(inputProps);
    const operation = new GPUSpaceTimeDiggleTestOperation(props);
    const {id} = operation;
    validateCapacity(id, 'rowCount', props.rowCount);
    validateOptions(id, props);
    resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'area'
    });
    if (props.temporalContext)
      resolveGPUTemporalContext(id, props.temporalContext, {encodings: ['float32-relative']});
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
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must match rowCount`);
        const result = computeSpaceTimeDiggleTestOnCPU(
          {
            positions,
            times: getCPUOperationFloat32Rows(id, inputs, 'times'),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            randomParameters: getCPUOperationUint32Rows(id, inputs, 'randomParameters'),
            mask:
              props.mask === undefined ? undefined : getCPUOperationUint32Rows(id, inputs, 'mask')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUSpaceTimeDiggleTestResult]
          ])
        );
      }
    });
  }
}
