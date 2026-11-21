// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationUint32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getChiSquareSurvival,
  getLogGammaOfHalfInteger
} from '../spatial-regression/ordinary-least-squares-statistics';
import {
  GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS,
  GPU_TRANSITION_HOMOGENEITY,
  GPU_TRANSITION_HOMOGENEITY_LENGTH,
  GPU_TRANSITION_STATE_SPARSITY,
  GPU_TRANSITION_STATE_SPARSITY_LENGTH
} from './gpu-transition-diagnostics';
import {
  captureTransitionDiagnosticsOperationProps,
  getTransitionDiagnosticsOperationProps,
  type TransitionDiagnosticsOperationProps
} from './gpu-transition-diagnostics-operation';

export type CPUTransitionDiagnosticsOptions = Pick<
  TransitionDiagnosticsOperationProps,
  'classCount' | 'conditionCount' | 'minimumTransitions'
>;
export type CPUTransitionDiagnosticsResult = {
  homogeneity: Float32Array;
  steadyStates: Float32Array;
  stateSparsity: Uint32Array;
  converged: Uint32Array;
};

/** Integer-df gamma recurrence from the public df=1 and df=2 survival functions, without a series cap. */
function getTransitionChiSquareSurvival(value: number, degreesOfFreedom: number): number {
  if (!(degreesOfFreedom > 0)) return NaN;
  if (!(value > 0)) return Number.isNaN(value) ? NaN : 1;
  if (!Number.isFinite(value)) return 0;
  const baseDegrees = degreesOfFreedom % 2 === 0 ? 2 : 1;
  const baseShape = baseDegrees / 2;
  const argument = value / 2;
  let survival = getChiSquareSurvival(value, baseDegrees);
  // Q(a+1,x) = Q(a,x) + exp(-x) x^a / Gamma(a+1).
  // Log terms avoid premature underflow before reaching the mode of a long recurrence.
  let logTerm =
    -argument +
    baseShape * Math.log(argument) -
    getLogGammaOfHalfInteger(baseDegrees) -
    Math.log(baseShape);
  for (let shape = baseShape; shape < degreesOfFreedom / 2; shape++) {
    survival += Math.exp(logTerm);
    logTerm += Math.log(argument) - Math.log(shape + 1);
  }
  return Math.min(1, Math.max(0, survival));
}

/**
 * Computes homogeneity, stationary distributions and sparsity in the GPU's scalar row layouts.
 * Stationarity checks strong connectivity on the count support, then uses float64 GTH elimination.
 * Empty chains return NaNs without clearing `converged`; any failed nonempty chain clears it.
 * Extra counts after conditionCount * classCount² are unused capacity, as on the GPU.
 *
 * What differs from the GPU contributor:
 * - Uses float64 sums and elimination, rounding each float output once; ordinary inputs should
 *   agree within 1e-5 relative for statistics and 2e-6 absolute for stationary probabilities.
 * - Uses the public float64 chi-square survival function at df=1 or 2 and a finite integer-df
 *   recurrence, without the GPU's incomplete-gamma iteration cap or large-df normal approximation.
 *   GPU p-values allow 2e-4 absolute error (including its large-df approximation).
 * - Drops the 64-class private-array limit; output layout imposes no corresponding class limit.
 * - Aggregated counts use float64 rather than float32; individual input and output words stay uint32.
 * - minimumTransitions is not capped at uint32, unlike the GPU's integer sparsity diagnostic.
 * - Rescales stationary weights during back-substitution, staying finite where the GPU's unscaled
 *   float32 weights overflow. A non-finite or non-positive normalization leaves the chain NaN and
 *   sets converged to 0; the GPU also clears converged if its unscaled weights overflow.
 */
export function computeTransitionDiagnosticsOnCPU(
  counts: Uint32Array,
  options: CPUTransitionDiagnosticsOptions
): CPUTransitionDiagnosticsResult {
  const {classCount} = options;
  const conditionCount = options.conditionCount ?? 1;
  const minimumTransitions =
    options.minimumTransitions ?? GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS;
  validateCapacity('computeTransitionDiagnosticsOnCPU', 'classCount', classCount);
  validateCapacity('computeTransitionDiagnosticsOnCPU', 'conditionCount', conditionCount);
  validateCapacity(
    'computeTransitionDiagnosticsOnCPU',
    'minimumTransitions',
    minimumTransitions,
    1,
    Number.MAX_SAFE_INTEGER
  );
  if (counts.length < conditionCount * classCount * classCount)
    throw new Error(
      'computeTransitionDiagnosticsOnCPU counts length must cover conditionCount * classCount * classCount'
    );
  const matrixLength = classCount * classCount;
  const chainCount = conditionCount > 1 ? conditionCount + 1 : 1;
  const pooled = new Float64Array(matrixLength);
  const pooledIntegers = new Array<bigint>(matrixLength).fill(0n);
  const stateSparsity = new Uint32Array(GPU_TRANSITION_STATE_SPARSITY_LENGTH);
  for (let condition = 0; condition < conditionCount; condition++) {
    for (let origin = 0; origin < classCount; origin++) {
      let total = 0;
      for (let destination = 0; destination < classCount; destination++) {
        const cell = origin * classCount + destination;
        const count = counts[condition * matrixLength + cell];
        pooled[cell] += count;
        pooledIntegers[cell] += BigInt(count);
        total += count;
      }
      if (total === 0) stateSparsity[GPU_TRANSITION_STATE_SPARSITY.EMPTY]++;
      else if (total < minimumTransitions) stateSparsity[GPU_TRANSITION_STATE_SPARSITY.SPARSE]++;
    }
  }
  let statistic = 0;
  let likelihoodRatio = 0;
  let degreesOfFreedom = 0;
  for (let origin = 0; origin < classCount; origin++) {
    let pooledTotalInteger = 0n;
    let nonzeroCells = 0;
    for (let destination = 0; destination < classCount; destination++) {
      const count = pooled[origin * classCount + destination];
      pooledTotalInteger += pooledIntegers[origin * classCount + destination];
      if (count > 0) nonzeroCells++;
    }
    const pooledTotal = Number(pooledTotalInteger);
    if (pooledTotal === 0) continue;
    let regimes = 0;
    for (let condition = 0; condition < conditionCount; condition++) {
      const start = condition * matrixLength + origin * classCount;
      let totalInteger = 0n;
      for (let destination = 0; destination < classCount; destination++)
        totalInteger += BigInt(counts[start + destination]);
      const total = Number(totalInteger);
      if (total === 0) continue;
      regimes++;
      for (let destination = 0; destination < classCount; destination++) {
        const observed = counts[start + destination];
        const expectedProduct = total * pooled[origin * classCount + destination];
        const expected = expectedProduct / pooledTotal;
        // Preserve the exact integer residual before converting: both binary64 products
        // can round to the same value even when their difference is one.
        const difference =
          Number(
            BigInt(observed) * pooledTotalInteger -
              totalInteger * pooledIntegers[origin * classCount + destination]
          ) / pooledTotal;
        if (expected > 0) {
          statistic += difference ** 2 / expected;
          const delta = difference / expected;
          if (Math.abs(delta) < 0.25) {
            let term = (delta * delta) / 2;
            let entropy = term;
            for (let order = 3; order <= 40; order++) {
              term *= (-delta * (order - 2)) / order;
              entropy += term;
            }
            likelihoodRatio += expected * entropy;
          } else {
            likelihoodRatio +=
              observed > 0 ? observed * Math.log(observed / expected) - difference : expected;
          }
        }
      }
    }
    degreesOfFreedom += (regimes - 1) * (nonzeroCells - 1);
  }
  likelihoodRatio = Math.max(0, 2 * likelihoodRatio);
  const homogeneity = new Float32Array(GPU_TRANSITION_HOMOGENEITY_LENGTH);
  homogeneity[GPU_TRANSITION_HOMOGENEITY.Q] = statistic;
  homogeneity[GPU_TRANSITION_HOMOGENEITY.Q_P_VALUE] = getTransitionChiSquareSurvival(
    statistic,
    degreesOfFreedom
  );
  homogeneity[GPU_TRANSITION_HOMOGENEITY.LR] = likelihoodRatio;
  homogeneity[GPU_TRANSITION_HOMOGENEITY.LR_P_VALUE] = getTransitionChiSquareSurvival(
    likelihoodRatio,
    degreesOfFreedom
  );
  homogeneity[GPU_TRANSITION_HOMOGENEITY.DEGREES_OF_FREEDOM] = degreesOfFreedom;
  const steadyStates = new Float32Array(chainCount * classCount).fill(NaN);
  const converged = Uint32Array.of(1);
  for (let chain = 0; chain < chainCount; chain++) {
    const matrix =
      chain === 0 ? pooled : counts.subarray((chain - 1) * matrixLength, chain * matrixLength);
    const active: number[] = [];
    const totals = new Float64Array(classCount);
    let solvable = true;
    for (let state = 0; state < classCount; state++) {
      let incoming = 0;
      for (let other = 0; other < classCount; other++) {
        totals[state] += matrix[state * classCount + other];
        incoming += matrix[other * classCount + state];
      }
      if (totals[state] > 0 || incoming > 0) {
        active.push(state);
        if (totals[state] === 0) solvable = false;
      }
    }
    const size = active.length;
    if (size === 0) continue;
    // Positive GTH escape sums alone do not prove irreducibility: a transient state can
    // escape to a lower-numbered closed class. Every active state must both reach and be
    // reachable from active[0]. Test the integer support, independent of numeric underflow.
    for (const reversed of [false, true]) {
      const reachable = new Uint8Array(size);
      reachable[0] = 1;
      for (let pass = 0; pass < size; pass++) {
        let changed = false;
        for (let origin = 0; origin < size; origin++) {
          if (!reachable[origin]) continue;
          for (let destination = 0; destination < size; destination++) {
            const cell = reversed
              ? active[destination] * classCount + active[origin]
              : active[origin] * classCount + active[destination];
            if (matrix[cell] > 0 && !reachable[destination]) {
              reachable[destination] = 1;
              changed = true;
            }
          }
        }
        if (!changed) break;
      }
      if (reachable.some(value => value === 0)) solvable = false;
    }
    const work = Float64Array.from({length: size * size}, (_, index) => {
      const origin = active[Math.floor(index / size)];
      return matrix[origin * classCount + active[index % size]] / totals[origin];
    });
    for (let eliminated = size - 1; solvable && eliminated >= 1; eliminated--) {
      let escapeProbability = 0;
      for (let destination = 0; destination < eliminated; destination++)
        escapeProbability += work[eliminated * size + destination];
      if (!(escapeProbability > 0)) {
        solvable = false;
        break;
      }
      for (let origin = 0; origin < eliminated; origin++)
        work[origin * size + eliminated] /= escapeProbability;
      for (let origin = 0; origin < eliminated; origin++) {
        for (let destination = 0; destination < eliminated; destination++)
          work[origin * size + destination] +=
            work[origin * size + eliminated] * work[eliminated * size + destination];
      }
    }
    if (!solvable) {
      converged[0] = 0;
      continue;
    }
    const solution = new Float64Array(size);
    solution[0] = 1;
    for (let state = 1; state < size; state++) {
      let weight = 0;
      for (let other = 0; other < state; other++)
        weight += solution[other] * work[other * size + state];
      // Keep every solved prefix normalized: GTH weights are homogeneous, so a common scale
      // preserves all subsequent ratios. Scale before summing to avoid overflowing 1 + weight.
      const scale = Math.max(1, weight);
      solution[state] = weight / scale;
      let normalization = solution[state];
      for (let other = 0; other < state; other++) {
        solution[other] /= scale;
        normalization += solution[other];
      }
      if (!Number.isFinite(normalization) || !(normalization > 0)) {
        solvable = false;
        break;
      }
      for (let other = 0; other <= state; other++) solution[other] /= normalization;
    }
    const total = solution.reduce((sum, weight) => sum + weight, 0);
    if (!solvable || !Number.isFinite(total) || !(total > 0) || !solution.every(Number.isFinite)) {
      converged[0] = 0;
      continue;
    }
    steadyStates.fill(0, chain * classCount, (chain + 1) * classCount);
    active.forEach((state, index) => {
      steadyStates[chain * classCount + state] = solution[index] / total;
    });
  }
  return {homogeneity, steadyStates, stateSparsity, converged};
}

/** CPU twin. Class count is not capped at the GPU private-array size of 64. */
export class CPUTransitionDiagnosticsOperation extends GPUCPUOperation {
  constructor(props: TransitionDiagnosticsOperationProps) {
    props = captureTransitionDiagnosticsOperationProps(props);
    const id = props.id ?? 'transition-diagnostics';
    const conditionCount = props.conditionCount ?? 1;
    const minimumTransitions =
      props.minimumTransitions ?? GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS;
    validateCapacity(id, 'classCount', props.classCount);
    validateCapacity(id, 'conditionCount', conditionCount);
    validateCapacity(id, 'minimumTransitions', minimumTransitions, 1, Number.MAX_SAFE_INTEGER);
    const {output} = props;
    validateRequirement(
      id,
      !output.homogeneity || conditionCount >= 2,
      'output.homogeneity needs conditionCount of at least 2'
    );
    validateRequirement(
      id,
      !output.status?.converged || Boolean(output.steadyStates),
      'output.status.converged needs output.steadyStates'
    );
    const chainCount = conditionCount > 1 ? conditionCount + 1 : 1;
    // Reuse the graph-port shape and disjointness checks; this allocates no GPU resources.
    new GPUContributorOperation(getTransitionDiagnosticsOperationProps(props));
    super({
      id,
      inputs: {counts: props.counts},
      outputs: {
        ...(output.homogeneity
          ? {
              [output.homogeneity]: gpuGraph.vector('float32', {
                length: GPU_TRANSITION_HOMOGENEITY_LENGTH
              })
            }
          : {}),
        ...(output.steadyStates
          ? {
              [output.steadyStates]: gpuGraph.vector('float32', {
                length: chainCount * props.classCount
              })
            }
          : {}),
        ...(output.stateSparsity
          ? {
              [output.stateSparsity]: gpuGraph.vector('uint32', {
                length: GPU_TRANSITION_STATE_SPARSITY_LENGTH
              })
            }
          : {}),
        ...(output.status?.converged
          ? {[output.status.converged]: gpuGraph.vector('uint32', {length: 1})}
          : {})
      },
      execute({inputs}) {
        const result = computeTransitionDiagnosticsOnCPU(
          getCPUOperationUint32Rows(id, inputs, 'counts'),
          props
        );
        return {
          ...(output.homogeneity ? {[output.homogeneity]: result.homogeneity} : {}),
          ...(output.steadyStates ? {[output.steadyStates]: result.steadyStates} : {}),
          ...(output.stateSparsity ? {[output.stateSparsity]: result.stateSparsity} : {}),
          ...(output.status?.converged ? {[output.status.converged]: result.converged} : {})
        };
      }
    });
  }
}
