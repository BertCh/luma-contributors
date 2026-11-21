// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  captureEnsembleScoreOutputs,
  resolveEnsembleScoresEstimator,
  type EnsembleScoresEstimatorOptions,
  type EnsembleScore
} from './ensemble-scores-options';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUEnsembleScoresOperation,
  type EnsembleScoresOperationProps
} from './gpu-ensemble-scores-operation';

export type CPUEnsembleScoresInput = {
  ensemble: Float32Array;
  observations: Float32Array;
  variogramPairs?: Uint32Array;
  variogramWeights?: Float32Array;
};

export type CPUEnsembleScoresOptions = EnsembleScoresEstimatorOptions & {
  memberCount: number;
  dimension: number;
  variogramOrder?: number;
  /** Defaults to CRPS, energy, and (when pairs exist) variogram. Only requested work is done. */
  scores?: readonly EnsembleScore[];
};

export type CPUEnsembleScoresResult = {
  crps?: Float32Array;
  energyScore?: Float32Array;
  variogramScore?: Float32Array;
};

function validateScoreOptions(id: string, options: CPUEnsembleScoresOptions): void {
  const estimator = resolveEnsembleScoresEstimator(id, options);
  validateCapacity(id, 'memberCount', options.memberCount, estimator === 'fair' ? 2 : 1);
  validateCapacity(id, 'dimension', options.dimension, 1);
  validateCapacity(id, 'memberCount * dimension', options.memberCount * options.dimension, 1);
  validateFiniteNumber(id, 'variogramOrder', options.variogramOrder ?? 0.5, 0);
  validateRequirement(id, (options.variogramOrder ?? 0.5) > 0, 'variogramOrder must be positive');
  if (options.scores)
    validateRequirement(
      id,
      options.scores.length > 0 &&
        options.scores.every(score => ['crps', 'energyScore', 'variogramScore'].includes(score)),
      'scores must request crps, energyScore or variogramScore'
    );
}

type EnsembleScoreTupleGuarantees<
  Scores extends readonly EnsembleScore[],
  Score extends EnsembleScore
> = Scores extends readonly [infer First, ...infer Rest extends readonly EnsembleScore[]]
  ? [First] extends [Score]
    ? true
    : EnsembleScoreTupleGuarantees<Rest, Score>
  : false;

type EnsembleScoreSelectionsMissing<
  Scores extends readonly EnsembleScore[],
  Score extends EnsembleScore
> = Scores extends unknown
  ? EnsembleScoreTupleGuarantees<Scores, Score> extends true
    ? never
    : Scores
  : never;

type GuaranteedEnsembleScores<Scores extends readonly EnsembleScore[]> = {
  [Score in Scores[number]]: [EnsembleScoreSelectionsMissing<Scores, Score>] extends [never]
    ? Score
    : never;
}[Scores[number]];

type SelectedEnsembleScoreFields<Scores extends readonly EnsembleScore[]> = {
  [Score in GuaranteedEnsembleScores<Scores>]: Float32Array;
} & {
  [Score in Exclude<Scores[number], GuaranteedEnsembleScores<Scores>>]?: Float32Array;
};

/** Require a key only when every possible tuple selection contains that exact score. */
export type CPUEnsembleScoresSelectedResult<Scores extends readonly EnsembleScore[]> =
  number extends Scores['length']
    ? CPUEnsembleScoresResult
    : {
        [Score in keyof SelectedEnsembleScoreFields<Scores>]: SelectedEnsembleScoreFields<Scores>[Score];
      };

/**
 * Scores a member-major ensemble in float64. The sorted gap identity computes the CRPS pair
 * term (or nonnegative fair intervals) in O(m log m) per site; energy uses unordered member pairs and variograms use listed pairs.
 *
 * What differs from the GPU contributor:
 * - All arithmetic uses float64 with one float32 output rounding; relative tolerance 3e-5 is
 *   appropriate for well-conditioned scores. Near cancellation may need an absolute error bound.
 * - GPU workgroup/invocation limits (65535 members and 67108863 sites) are dropped; uint32 array
 *   indexing limits remain. Unrequested scores are not computed.
 * - Non-finite ensemble members give NaN CRPS/energy; non-finite observations follow IEEE
 *   arithmetic. Variograms give NaN for out-of-range pairs or negative/nonfinite weights.
 * - Subnormals and finite intermediates beyond float32 range are retained in float64. The plain
 *   numeric variogramOrder prop is evaluated in float64, rather than rounded in a shader literal.
 *
 * @throws If ensemble/observations are too short, pairs are incomplete, or weights are too short.
 */
export function computeEnsembleScoresOnCPU<const Scores extends readonly EnsembleScore[]>(
  input: CPUEnsembleScoresInput,
  options: CPUEnsembleScoresOptions & {scores: Scores}
): CPUEnsembleScoresSelectedResult<Scores>;
export function computeEnsembleScoresOnCPU(
  input: CPUEnsembleScoresInput,
  options: CPUEnsembleScoresOptions
): CPUEnsembleScoresResult;
export function computeEnsembleScoresOnCPU(
  input: CPUEnsembleScoresInput,
  options: CPUEnsembleScoresOptions
): CPUEnsembleScoresResult {
  validateScoreOptions('computeEnsembleScoresOnCPU', options);
  const {ensemble, observations, variogramPairs, variogramWeights} = input;
  const {memberCount, dimension} = options;
  if (ensemble.length < memberCount * dimension)
    throw new Error('ensemble needs memberCount * dimension values');
  if (observations.length < dimension) throw new Error('observations needs dimension values');
  if (variogramPairs && variogramPairs.length % 2 !== 0)
    throw new Error('variogramPairs must contain complete pairs');
  if (variogramWeights && (!variogramPairs || variogramWeights.length < variogramPairs.length / 2))
    throw new Error('variogramWeights needs one value per variogramPairs row');
  const scores = options.scores ?? [
    'crps',
    'energyScore',
    ...(variogramPairs ? ['variogramScore' as const] : [])
  ];
  if (scores.includes('variogramScore') && (!variogramPairs || variogramPairs.length === 0))
    throw new Error('variogramPairs needs at least one pair for variogramScore');
  const result: CPUEnsembleScoresResult = {};
  const fair = resolveEnsembleScoresEstimator('computeEnsembleScoresOnCPU', options) === 'fair';
  const pairDenominator = memberCount * (fair ? memberCount - 1 : memberCount);
  let scalarScore: number | undefined;
  if (scores.includes('crps') || (fair && dimension === 1 && scores.includes('energyScore'))) {
    const crps = new Float32Array(dimension);
    for (let site = 0; site < dimension; site++) {
      const sorted: number[] = [];
      let observationSum = 0;
      for (let member = 0; member < memberCount; member++) {
        const value = ensemble[member * dimension + site];
        sorted.push(value);
        if (!fair) observationSum += Math.abs(value - observations[site]);
      }
      if (!sorted.every(Number.isFinite)) {
        crps[site] = NaN;
        continue;
      }
      sorted.sort((first, second) => first - second);
      let pairSum = 0;
      for (let member = 1; !fair && member < memberCount; member++)
        pairSum += (sorted[member] - sorted[member - 1]) * member * (memberCount - member);
      if (fair) {
        // Integrate the fair CDF score over sorted intervals. Below y, the coefficient is
        // k(k-1)/(m(m-1)); above y it is (m-k)(m-k-1)/(m(m-1)). All terms are nonnegative.
        const observed = observations[site];
        let total = Math.max(sorted[0] - observed, 0, observed - sorted[memberCount - 1]);
        for (let member = 1; member < memberCount; member++) {
          const lower = sorted[member - 1];
          const upper = sorted[member];
          if (observed > lower)
            total +=
              (Math.min(observed, upper) - lower) *
              ((member / memberCount) * ((member - 1) / (memberCount - 1)));
          if (observed < upper)
            total +=
              (upper - Math.max(observed, lower)) *
              (((memberCount - member) / memberCount) *
                ((memberCount - member - 1) / (memberCount - 1)));
        }
        crps[site] = total;
      } else {
        crps[site] = observationSum / memberCount - pairSum / pairDenominator;
      }
    }
    if (dimension === 1) scalarScore = crps[0];
    if (scores.includes('crps')) result.crps = crps;
  }
  if (scores.includes('energyScore') && fair && dimension === 1) {
    result.energyScore = Float32Array.of(scalarScore!);
  } else if (scores.includes('energyScore')) {
    let observationSum = 0;
    let pairSum = 0;
    for (let member = 0; member < memberCount; member++) {
      let observedSquares = 0;
      for (let site = 0; site < dimension; site++) {
        const difference = ensemble[member * dimension + site] - observations[site];
        observedSquares += difference * difference;
      }
      observationSum += Math.sqrt(observedSquares);
      for (let other = 0; other < member; other++) {
        let squares = 0;
        for (let site = 0; site < dimension; site++) {
          const difference =
            ensemble[member * dimension + site] - ensemble[other * dimension + site];
          squares += difference * difference;
        }
        pairSum += Math.sqrt(squares);
      }
    }
    // GPU self-pairs also turn a lone infinite member into NaN.
    const finiteEnsemble = ensemble.subarray(0, memberCount * dimension).every(Number.isFinite);
    result.energyScore = Float32Array.of(
      finiteEnsemble ? observationSum / memberCount - pairSum / pairDenominator : NaN
    );
  }
  if (scores.includes('variogramScore') && variogramPairs) {
    const order = options.variogramOrder ?? 0.5;
    let total = 0;
    for (let pair = 0; pair < variogramPairs.length / 2; pair++) {
      const first = variogramPairs[pair * 2];
      const second = variogramPairs[pair * 2 + 1];
      const weight = variogramWeights ? variogramWeights[pair] : 1;
      if (first >= dimension || second >= dimension || !Number.isFinite(weight) || weight < 0) {
        total = NaN;
        break;
      }
      const observedFirst = observations[first];
      const observedSecond = observations[second];
      let valid = Number.isFinite(observedFirst) && Number.isFinite(observedSecond);
      const observedGap = Math.abs(observedFirst - observedSecond);
      let maximumGap = observedGap;
      for (let member = 0; member < memberCount; member++) {
        const firstValue = ensemble[member * dimension + first];
        const secondValue = ensemble[member * dimension + second];
        valid = valid && Number.isFinite(firstValue) && Number.isFinite(secondValue);
        maximumGap = Math.max(maximumGap, Math.abs(firstValue - secondValue));
      }
      if (!valid) {
        total = NaN;
        break;
      }
      if (maximumGap === 0 || weight === 0) continue;
      const exponent = Math.floor(Math.log2(maximumGap));
      const scalePower = order === 1 ? exponent : Math.log2(maximumGap) * order;
      let discrepancySum = 0;
      for (let member = 0; member < memberCount; member++) {
        const forecastGap = Math.abs(
          ensemble[member * dimension + first] - ensemble[member * dimension + second]
        );
        if (order === 1) {
          // A power-of-two frame preserves exact small differences between large gaps.
          discrepancySum += (observedGap - forecastGap) / 2 ** exponent;
        } else if (order === 0.5 && observedGap !== forecastGap) {
          discrepancySum +=
            (observedGap - forecastGap) /
            (Math.sqrt(observedGap) + Math.sqrt(forecastGap)) /
            Math.sqrt(maximumGap);
        } else {
          discrepancySum +=
            (observedGap / maximumGap) ** order - (forecastGap / maximumGap) ** order;
        }
      }
      const difference = discrepancySum / memberCount;
      // Restore only the final weighted square. In particular, equal finite high-order gaps
      // never form Infinity - Infinity, and a perfect forecast remains exactly zero.
      if (difference !== 0)
        total += 2 ** (2 * (Math.log2(Math.abs(difference)) + scalePower) + Math.log2(weight));
    }
    result.variogramScore = Float32Array.of(total);
  }
  return result;
}

export type CPUEnsembleScoresOperationProps = EnsembleScoresOperationProps;

/** CPU twin; removes GPU invocation limits and computes only the requested scores. */
export class CPUEnsembleScoresOperation extends GPUCPUOperation {
  constructor(props: EnsembleScoresOperationProps) {
    const id = props.id ?? 'ensemble-scores';
    const scoreDestinations = captureEnsembleScoreOutputs(id, props.output);
    const scores = scoreDestinations.map(([score]) => score);
    const {variogramPairs, variogramWeights} = props;
    const options = {
      memberCount: props.memberCount,
      dimension: props.dimension,
      estimator: resolveEnsembleScoresEstimator(id, props),
      variogramOrder: props.variogramOrder,
      scores
    };
    validateScoreOptions(id, options);
    validateRequirement(
      id,
      (props.output.variogramScore !== undefined) === (props.variogramPairs !== undefined),
      'output.variogramScore and variogramPairs must go together'
    );
    validateRequirement(
      id,
      props.variogramWeights === undefined || props.variogramPairs !== undefined,
      'variogramWeights requires variogramPairs'
    );
    new GPUEnsembleScoresOperation(props);
    super({
      id,
      inputs: {
        ensemble: props.ensemble,
        observations: props.observations,
        ...(props.variogramPairs !== undefined ? {variogramPairs: props.variogramPairs} : {}),
        ...(props.variogramWeights !== undefined ? {variogramWeights: props.variogramWeights} : {})
      },
      outputs: Object.fromEntries(
        scoreDestinations.map(([score, destination]) => [
          destination,
          gpuGraph.vector('float32', {length: score === 'crps' ? props.dimension : 1})
        ])
      ),
      execute({inputs}) {
        const result = computeEnsembleScoresOnCPU(
          {
            ensemble: getCPUOperationFloat32Rows(id, inputs, 'ensemble'),
            observations: getCPUOperationFloat32Rows(id, inputs, 'observations'),
            variogramPairs:
              variogramPairs !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'variogramPairs', 'uint32x2')
                : undefined,
            variogramWeights:
              variogramWeights !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'variogramWeights')
                : undefined
          },
          options
        );
        return Object.fromEntries(
          scoreDestinations.map(([score, destination]) => [destination, result[score]!])
        );
      }
    });
  }
}
