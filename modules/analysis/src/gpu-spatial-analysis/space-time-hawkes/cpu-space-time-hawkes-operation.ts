// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {resolveGPUSpaceTimeTemporalContext} from '../space-time-tests/space-time-bands';
import {
  GPUSpaceTimeHawkesOperation,
  type SpaceTimeHawkesOperationProps
} from './gpu-space-time-hawkes-operation';
import {GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH} from './space-time-hawkes-parameters';
import {
  GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR,
  GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR
} from './space-time-hawkes-wgsl';

/** Finite CPU work budget: sub-machine-precision tolerances can leave EM in a rounding cycle. */
export const CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS = 1000;

/** Packed input vectors; parameters use getGPUSpaceTimeHawkesParameterValues' nine-value layout. */
export type CPUSpaceTimeHawkesInput = {
  positions: Float32Array;
  times: Float32Array;
  parameters: Float32Array;
};
/** CPU work budget; the GPU's 1000-iteration unrolling cap does not constrain an explicit budget. */
export type CPUSpaceTimeHawkesOptions = {
  maximumIterations?: number;
};
/** Every possible output, including the optional scalar status ports. */
export type CPUSpaceTimeHawkesResult = {
  estimates: Float32Array;
  logLikelihood: Float32Array;
  backgroundProbability: Float32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  iterationCount: Uint32Array;
  candidateOverflow: Uint32Array;
  invalidCount: Uint32Array;
  approximation: Uint32Array;
};

/**
 * Fits the documented exponential-Gaussian Hawkes update until its relative-change tolerance or
 * iteration budget is reached, using every earlier parent inside the inclusive lag/distance window.
 * Equal-time events do not trigger each other. The convergence comparison uses spatial variance,
 * as the kernel does.
 * logLikelihood is evaluated at the iterate entering the final update; probabilities use the
 * published estimates. The finite-window temporal mass is evaluated with expm1 in float64.
 *
 * What differs from the GPU contributor:
 * - No parent capacity; candidateOverflow is always zero. A finite CPU iteration budget defaults
 *   to 1000, versus the GPU's default 100. An explicit maximumIterations is honored by both
 *   operations; CPU accepts budgets above the GPU's 1000-iteration unrolling cap, up to uint32.
 *   At the budget, the latest fit is published with converged zero and iterationLimitReached one
 *   unless that update converged. This also terminates floating-point rounding cycles.
 * - Float64 log-domain evaluation and sums replace float32 tiles. Compare fitted values and
 *   probabilities at relative 5e-4; convergence counts can differ near the tolerance threshold.
 * - The exact -expm1(-beta * remainingTime) replaces the GPU's second-order small-argument mass.
 * - The GPU's log-rate floors, branching-ratio floor and 1e-6 triggered-mass update guard are
 *   retained. Spatial/temporal cutoff approximation remains, so approximation is still one.
 * - The GPU-only 2^24 event cap is dropped; the uint32 iteration/status layout limit remains.
 * - Nonfinite coordinates/times are excluded. Nonfinite iterates publish NaNs and stop with
 *   converged zero; subnormal float32 outputs are retained. These are outside GPU comparisons.
 *
 * @throws If positions/times lengths disagree, parameters has fewer than nine values, or
 * maximumIterations is not an integer in [1, 2^32 - 1].
 */
export function computeSpaceTimeHawkesOnCPU(
  input: CPUSpaceTimeHawkesInput,
  options: CPUSpaceTimeHawkesOptions = {}
): CPUSpaceTimeHawkesResult {
  const maximumIterations =
    options.maximumIterations ?? CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS;
  validateCapacity('computeSpaceTimeHawkesOnCPU', 'maximumIterations', maximumIterations);
  const {positions, times, parameters} = input;
  if (positions.length !== times.length * 2)
    throw new Error('positions length must equal times.length * 2');
  if (parameters.length < GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH)
    throw new Error(`parameters must hold ${GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH} values`);
  // Named destructuring follows the packer's documented layout; there are no exported offsets.
  const [
    area,
    duration,
    maximumLag,
    maximumDistance,
    tolerance,
    initialBackground,
    initialBranchingRatio,
    initialDecayRate,
    initialDeviation
  ] = parameters;
  const validParameters =
    [
      area,
      duration,
      maximumLag,
      maximumDistance,
      initialBackground,
      initialBranchingRatio,
      initialDecayRate,
      initialDeviation
    ].every(value => value > 0 && Number.isFinite(value)) &&
    tolerance > 0 &&
    Number.isFinite(tolerance);
  const rows = Array.from({length: times.length}, (_, row) => row)
    .filter(
      row =>
        Number.isFinite(positions[row * 2]) &&
        Number.isFinite(positions[row * 2 + 1]) &&
        Number.isFinite(times[row]) &&
        times[row] >= 0 &&
        times[row] <= duration
    )
    .sort((first, second) => times[first] - times[second] || first - second);
  let background = validParameters ? initialBackground : NaN;
  let branchingRatio = validParameters ? initialBranchingRatio : NaN;
  let decayRate = validParameters ? initialDecayRate : NaN;
  let variance = validParameters ? initialDeviation * initialDeviation : NaN;
  let iterations = 0;
  let converged = false;
  let iterationLimitReached = false;
  let logLikelihood = NaN;
  const evaluateEvent = (index: number) => {
    const row = rows[index];
    let maximumLog = Math.log(Math.max(background, GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR));
    const logProductivity =
      Math.log(Math.max(branchingRatio, GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR)) +
      Math.log(decayRate) -
      Math.log(2 * Math.PI * variance);
    let backgroundWeight = 1;
    let parentWeight = 0;
    let lagWeight = 0;
    let distanceWeight = 0;
    for (let parent = index - 1; parent >= 0; parent--) {
      const parentRow = rows[parent];
      const lag = times[row] - times[parentRow];
      if (times[parentRow] < times[row] - maximumLag) break;
      const differenceX = positions[row * 2] - positions[parentRow * 2];
      const differenceY = positions[row * 2 + 1] - positions[parentRow * 2 + 1];
      const distanceSquared = differenceX * differenceX + differenceY * differenceY;
      if (!(lag > 0) || distanceSquared > maximumDistance * maximumDistance) continue;
      const term = logProductivity - decayRate * lag - distanceSquared * (0.5 / variance);
      if (term > maximumLog) {
        const scale = Math.exp(maximumLog - term);
        backgroundWeight *= scale;
        parentWeight = parentWeight * scale + 1;
        lagWeight = lagWeight * scale + lag;
        distanceWeight = distanceWeight * scale + distanceSquared;
        maximumLog = term;
      } else {
        const weight = Math.exp(term - maximumLog);
        parentWeight += weight;
        lagWeight += weight * lag;
        distanceWeight += weight * distanceSquared;
      }
    }
    const sum = backgroundWeight + parentWeight;
    return {
      background: backgroundWeight / sum,
      triggered: parentWeight / sum,
      lag: lagWeight / sum,
      distance: distanceWeight / sum,
      logIntensity: maximumLog + Math.log(sum)
    };
  };
  if (validParameters) {
    while (!converged && iterations < maximumIterations) {
      iterations++;
      let backgroundSum = 0;
      let triggeredSum = 0;
      let lagSum = 0;
      let distanceSum = 0;
      let logSum = 0;
      let windowMass = 0;
      rows.forEach((row, index) => {
        const statistics = evaluateEvent(index);
        backgroundSum += statistics.background;
        triggeredSum += statistics.triggered;
        lagSum += statistics.lag;
        distanceSum += statistics.distance;
        logSum += statistics.logIntensity;
        windowMass += -Math.expm1(-decayRate * (duration - times[row]));
      });
      logLikelihood = logSum - (background * area * duration + branchingRatio * windowMass);
      if (rows.length === 0 || !(windowMass > 0)) {
        background = branchingRatio = decayRate = variance = NaN;
        break;
      }
      const nextBackground = backgroundSum / (area * duration);
      const nextBranchingRatio = Math.max(
        triggeredSum / windowMass,
        GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR
      );
      const updateKernel = triggeredSum > 1e-6 && lagSum > 0 && distanceSum > 0;
      const nextDecayRate = updateKernel ? triggeredSum / lagSum : decayRate;
      const nextVariance = updateKernel ? distanceSum / (2 * triggeredSum) : variance;
      const previous = [background, branchingRatio, decayRate, variance];
      const next = [nextBackground, nextBranchingRatio, nextDecayRate, nextVariance];
      const change = Math.max(
        ...next.map(
          (value, index) =>
            Math.abs(value - previous[index]) /
            Math.max(Math.abs(previous[index]), GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR)
        )
      );
      [background, branchingRatio, decayRate, variance] = next;
      if (!next.every(Number.isFinite)) {
        background = branchingRatio = decayRate = variance = NaN;
        break;
      }
      converged = change <= tolerance;
      iterationLimitReached = !converged && iterations === maximumIterations;
    }
  }
  const backgroundProbability = new Float32Array(times.length).fill(NaN);
  rows.forEach((row, index) => {
    backgroundProbability[row] = evaluateEvent(index).background;
  });
  return {
    estimates: Float32Array.of(background, branchingRatio, decayRate, Math.sqrt(variance)),
    logLikelihood: Float32Array.of(logLikelihood),
    backgroundProbability,
    converged: Uint32Array.of(Number(converged)),
    iterationLimitReached: Uint32Array.of(Number(iterationLimitReached)),
    iterationCount: Uint32Array.of(iterations),
    candidateOverflow: Uint32Array.of(0),
    invalidCount: Uint32Array.of(validParameters ? times.length - rows.length : times.length),
    approximation: Uint32Array.of(1)
  };
}

/** CPU fit; maximumParents specializes only the GPU operation. Both honor maximumIterations. */
export class CPUSpaceTimeHawkesOperation extends GPUCPUOperation {
  constructor(props: SpaceTimeHawkesOperationProps) {
    const id = props.id ?? 'space-time-hawkes';
    validateCapacity(id, 'rowCount', props.rowCount, 2);
    if (props.maximumIterations !== undefined)
      validateCapacity(id, 'maximumIterations', props.maximumIterations);
    if (props.maximumParents !== undefined)
      validateCapacity(id, 'maximumParents', props.maximumParents);
    resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'area'
    });
    resolveGPUSpaceTimeTemporalContext(id, props.temporalContext);
    const operation = new GPUSpaceTimeHawkesOperation(props);
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
        const times = getCPUOperationFloat32Rows(id, inputs, 'times');
        if (times.length !== props.rowCount)
          throw new Error(`${id} times length must equal rowCount`);
        const result = computeSpaceTimeHawkesOnCPU(
          {
            positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
            times,
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
          },
          {maximumIterations: props.maximumIterations}
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUSpaceTimeHawkesResult]
          ])
        );
      }
    });
  }
}
