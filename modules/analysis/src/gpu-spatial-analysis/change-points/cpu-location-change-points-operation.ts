// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  LOCATION_CHANGE_POINT_MEAN_SCALE_EXPONENT,
  LOCATION_CHANGE_POINT_MEAN_SCALE_THRESHOLD_EXPONENT,
  LOCATION_CHANGE_POINT_STANDARDIZATION_EXPONENT,
  LOCATION_CHANGE_POINT_STANDARDIZATION_THRESHOLD_EXPONENT,
  validateLocationChangeShape,
  validateLocationChangeOperationPorts
} from './change-point-shape';
import {snapshotChangePointProps} from './change-point-construction';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPU_LOCATION_CHANGE_POINTS_NONE,
  GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH
} from './gpu-location-change-points';
import {type LocationChangePointsOperationProps} from './gpu-location-change-points-operation';

export type CPULocationChangePointsOptions = Pick<
  LocationChangePointsOperationProps,
  'id' | 'cellCount' | 'sliceCount' | 'layout' | 'cost' | 'maximumChangePoints'
>;
export type CPULocationChangePointsResult = {
  changePointCount: Uint32Array;
  changePoints: Uint32Array;
  requiredChangePointCounts: Uint32Array;
  optimalCost: Float32Array;
  overflow: Uint32Array;
};

/** Same relative variance floor as the contributor, in data units after multiplying series variance. */
const VARIANCE_FLOOR = 1e-11;

// Poisson observed*log(observed/expected) - observed + expected. Near equal counts the
// entropy series converges until another term cannot change the float64 sum, rather than
// subtracting two almost equal logarithm terms or imposing the GPU polynomial's degree cap.
function computePoissonObservedTerm(observed: number, expected: number): number {
  if (!(observed > 0)) return expected;
  const difference = observed - expected;
  const delta = difference / expected;
  if (Math.abs(delta) < 0.25) {
    let term = (delta * delta) / 2;
    let entropy = term;
    for (let order = 3; ; order++) {
      term *= (-delta * (order - 2)) / order;
      const nextEntropy = entropy + term;
      if (nextEntropy === entropy) return expected * entropy;
      entropy = nextEntropy;
    }
  }
  return observed * Math.log(observed / expected) - difference;
}

/**
 * Exact unpruned optimal partitioning of each finite series, using float64 running central
 * moments, true slice positions and the contributor's relative variance floor. Starts are visited
 * in ascending order; a strict improvement keeps the smallest preceding start on exact ties.
 * The earliest changes are kept on overflow, while required counts and optimum remain unclamped.
 *
 * What differs from the GPU contributor:
 * - Costs and partition comparisons use float64 with one output rounding. Near ties may choose
 *   different partitions; compare well-separated optima with relative cost tolerance 3e-4.
 * - The GPU's 512-slice, 256-slope-slice and 128-change capacity limits are dropped; uint32 shape
 *   and index limits remain. The full quadratic dynamic program is still evaluated.
 * - Normal costs use float64 central moments. Wide variance, mean-variance and slope inputs
 *   are rescaled before centering and their costs are restored to data units.
 *   Poisson partitioning uses centered deviance and a float64 entropy series without the GPU's
 *   degree cap; the common likelihood constant is restored only for optimalCost. The relative
 *   variance floor and constant-series special cases are preserved. Both forms evaluate the
 *   first segment without subtracting and re-adding the penalty.
 * - Negative Poisson observations (outside the documented nonnegative-count domain) yield no
 *   changes and NaN cost. Invalid packed penalty/segment-length parameters throw naming parameters.
 * - Subnormals are retained, and finite float64 intermediate costs may exceed float32 range;
 *   only the published optimalCost is rounded, potentially to infinity.
 *
 * @throws If values does not exactly match the cube dimensions or parameters is malformed.
 */
export function computeLocationChangePointsOnCPU(
  values: Float32Array,
  parameters: Float32Array,
  options: CPULocationChangePointsOptions
): CPULocationChangePointsResult {
  options = snapshotConstructionFields(options, [
    'id',
    'cellCount',
    'sliceCount',
    'layout',
    'cost',
    'maximumChangePoints'
  ]);
  const id = options.id ?? 'computeLocationChangePointsOnCPU';
  validateLocationChangeShape(id, options, 'cpu');
  const {cellCount, sliceCount, layout, cost, maximumChangePoints} = options;
  validateRequirement(
    id,
    values.length === cellCount * sliceCount,
    `values length must equal cellCount * sliceCount (${cellCount * sliceCount}); received ${values.length}`
  );
  validateRequirement(
    id,
    parameters.length >= GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH,
    `parameters length must hold penalty and minimumSegmentLength (${GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH}); received ${parameters.length}`
  );
  const [penalty, packedMinimumLength] = parameters;
  validateRequirement(
    id,
    Number.isFinite(penalty) && penalty >= 0 && Number.isFinite(packedMinimumLength),
    `parameters requires a finite nonnegative penalty and finite minimumSegmentLength; received [${penalty}, ${packedMinimumLength}]`
  );
  const minimumLength = Math.max(1, Math.min(0xffffffff, Math.trunc(packedMinimumLength)));
  const usesStandardization = cost === 'variance' || cost === 'mean-variance' || cost === 'slope';
  const result: CPULocationChangePointsResult = {
    changePointCount: new Uint32Array(cellCount),
    changePoints: new Uint32Array(cellCount * maximumChangePoints).fill(
      GPU_LOCATION_CHANGE_POINTS_NONE
    ),
    requiredChangePointCounts: new Uint32Array(cellCount),
    optimalCost: new Float32Array(cellCount).fill(NaN),
    overflow: Uint32Array.of(0)
  };
  for (let cell = 0; cell < cellCount; cell++) {
    const series: number[] = [];
    const positions: number[] = [];
    let maximumMagnitude = 0;
    for (let slice = 0; slice < sliceCount; slice++) {
      const value =
        values[layout === 'cell-major' ? cell * sliceCount + slice : slice * cellCount + cell];
      if (Number.isFinite(value)) {
        series.push(value);
        positions.push(slice);
        if (usesStandardization) maximumMagnitude = Math.max(maximumMagnitude, Math.abs(value));
      }
    }
    const length = series.length;
    if (length < minimumLength || (cost === 'poisson' && series.some(value => value < 0))) continue;
    const standardizationShift =
      maximumMagnitude >= 2 ** LOCATION_CHANGE_POINT_STANDARDIZATION_THRESHOLD_EXPONENT
        ? Math.floor(Math.log2(maximumMagnitude)) - LOCATION_CHANGE_POINT_STANDARDIZATION_EXPONENT
        : 0;
    if (standardizationShift > 0) {
      const valueScale = 2 ** -standardizationShift;
      for (let index = 0; index < length; index++) series[index] *= valueScale;
    }
    const squareUnit = 2 ** (2 * standardizationShift);
    const logSquareUnit = 2 * standardizationShift * Math.LN2;
    // Center about the first value before averaging to preserve small changes on a large baseline.
    let centeredTotal = 0;
    let total = 0;
    for (const value of series) {
      centeredTotal += value - series[0];
      total += value;
    }
    const meanOffset = centeredTotal / length;
    let overallSquares = 0;
    for (const value of series) overallSquares += (value - series[0] - meanOffset) ** 2;
    const variance = overallSquares / length;
    const overallRate = series[0] + meanOffset;
    const hasStructure = length >= 2 && (cost === 'poisson' ? total > 0 : variance > 0);
    if (!hasStructure) {
      result.optimalCost[cell] =
        cost === 'variance' || cost === 'mean-variance'
          ? length * (Math.log(2 * Math.PI) + Math.log(VARIANCE_FLOOR) + 1)
          : cost === 'poisson' && total > 0
            ? -2 * total * Math.log(overallRate)
            : 0;
      continue;
    }
    const best = new Float64Array(length + 1);
    const previous = new Float64Array(length + 1).fill(-1);
    best[0] = 0;
    for (let start = 0; start < length; start++) {
      if (start > 0 && previous[start] < 0) continue;
      let segmentMean = 0;
      let centralSquares = 0;
      let meanMomentScale = 1;
      let aboutOverallSquares = 0;
      let segmentTotal = 0;
      let timeMean = 0;
      let timeSquares = 0;
      let covariance = 0;
      for (let end = start + 1; end <= length; end++) {
        const segmentLength = end - start;
        const value = series[end - 1];
        let offsetValue = value - series[start];
        if (
          cost === 'mean' &&
          meanMomentScale === 1 &&
          Math.abs(offsetValue) >= 2 ** LOCATION_CHANGE_POINT_MEAN_SCALE_THRESHOLD_EXPONENT
        ) {
          // Rebase the running central moments at the same power-of-two threshold as the GPU.
          meanMomentScale = 2 ** -LOCATION_CHANGE_POINT_MEAN_SCALE_EXPONENT;
          segmentMean *= meanMomentScale;
          centralSquares *= meanMomentScale ** 2;
        }
        offsetValue *= meanMomentScale;
        const difference = offsetValue - segmentMean;
        segmentMean += difference / segmentLength;
        centralSquares += difference * (offsetValue - segmentMean);
        aboutOverallSquares += (value - series[0] - meanOffset) ** 2;
        segmentTotal += value;
        const time = positions[end - 1] - positions[start];
        const timeDifference = time - timeMean;
        timeMean += timeDifference / segmentLength;
        timeSquares += timeDifference * (time - timeMean);
        covariance += timeDifference * (offsetValue - segmentMean);
        if (segmentLength < minimumLength) continue;
        let segmentCost: number;
        switch (cost) {
          case 'mean':
            segmentCost = Math.max(0, centralSquares / meanMomentScale ** 2);
            break;
          case 'slope':
            segmentCost =
              Math.max(
                0,
                centralSquares - (timeSquares > 0 ? (covariance * covariance) / timeSquares : 0)
              ) * squareUnit;
            break;
          case 'poisson':
            segmentCost =
              -2 * computePoissonObservedTerm(segmentTotal, segmentLength * overallRate);
            break;
          default: {
            const segmentVariance =
              (cost === 'variance' ? aboutOverallSquares : centralSquares) / segmentLength;
            segmentCost =
              segmentLength *
              (Math.log(2 * Math.PI) +
                Math.log(Math.max(segmentVariance, VARIANCE_FLOOR * variance)) +
                logSquareUnit +
                1);
          }
        }
        const candidate = best[start] + segmentCost + (start > 0 ? penalty : 0);
        if (previous[end] < 0 || candidate < best[end]) {
          best[end] = candidate;
          previous[end] = start;
        }
      }
    }
    if (previous[length] < 0) continue;
    const changes: number[] = [];
    for (let cursor = length; previous[cursor] > 0; cursor = previous[cursor])
      changes.push(positions[previous[cursor]]);
    changes.reverse();
    const kept = Math.min(changes.length, maximumChangePoints);
    result.changePointCount[cell] = kept;
    result.requiredChangePointCounts[cell] = changes.length;
    result.changePoints.set(changes.slice(0, kept), cell * maximumChangePoints);
    result.optimalCost[cell] =
      cost === 'poisson' ? best[length] - 2 * total * Math.log(overallRate) : best[length];
    if (changes.length > maximumChangePoints) result.overflow[0] = 1;
  }
  return result;
}

/** CPU twin; full partitioning without the GPU private-array and integer-time limits. */
export class CPULocationChangePointsOperation extends GPUCPUOperation {
  constructor(props: LocationChangePointsOperationProps) {
    props = snapshotChangePointProps(props);
    const id = props.id ?? 'location-change-points';
    validateLocationChangeShape(id, props, 'cpu');
    validateLocationChangeOperationPorts(id, props);
    super({
      id,
      inputs: {values: props.values, parameters: props.parameters},
      outputs: {
        [props.output.changePointCount]: gpuGraph.vector('uint32', {length: props.cellCount}),
        [props.output.changePoints]: gpuGraph.vector('uint32', {
          length: props.cellCount * props.maximumChangePoints
        }),
        ...(props.output.requiredChangePointCounts !== undefined
          ? {
              [props.output.requiredChangePointCounts]: gpuGraph.vector('uint32', {
                length: props.cellCount
              })
            }
          : {}),
        ...(props.output.optimalCost !== undefined
          ? {[props.output.optimalCost]: gpuGraph.vector('float32', {length: props.cellCount})}
          : {}),
        ...(props.output.status !== undefined
          ? {[props.output.status.overflow]: gpuGraph.vector('uint32', {length: 1})}
          : {})
      },
      execute({inputs}) {
        const result = computeLocationChangePointsOnCPU(
          getCPUOperationFloat32Rows(id, inputs, 'values'),
          getCPUOperationFloat32Rows(id, inputs, 'parameters'),
          props
        );
        return {
          [props.output.changePointCount]: result.changePointCount,
          [props.output.changePoints]: result.changePoints,
          ...(props.output.requiredChangePointCounts !== undefined
            ? {[props.output.requiredChangePointCounts]: result.requiredChangePointCounts}
            : {}),
          ...(props.output.optimalCost !== undefined
            ? {[props.output.optimalCost]: result.optimalCost}
            : {}),
          ...(props.output.status !== undefined
            ? {[props.output.status.overflow]: result.overflow}
            : {})
        };
      }
    });
  }
}
