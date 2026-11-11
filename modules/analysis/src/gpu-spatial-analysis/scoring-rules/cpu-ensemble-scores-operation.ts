// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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

export type CPUEnsembleScoresOptions = {
  memberCount: number;
  dimension: number;
  fair?: boolean;
  variogramOrder?: number;
  /** Defaults to CRPS, energy, and (when pairs exist) variogram. Only requested work is done. */
  scores?: readonly ('crps' | 'energyScore' | 'variogramScore')[];
};

export type CPUEnsembleScoresResult = {
  crps?: Float32Array;
  energyScore?: Float32Array;
  variogramScore?: Float32Array;
};

function validateScoreOptions(id: string, options: CPUEnsembleScoresOptions): void {
  validateCapacity(id, 'memberCount', options.memberCount, options.fair ? 2 : 1);
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

/**
 * Scores a member-major ensemble in float64. The sorted gap identity computes the CRPS pair
 * term in O(m log m) per site; energy uses unordered member pairs and variograms use listed pairs.
 *
 * What differs from the GPU contributor:
 * - All arithmetic uses float64 with one float32 output rounding; relative tolerance 3e-5 is
 *   appropriate for well-conditioned scores. Near cancellation may need an absolute error bound.
 * - GPU workgroup/invocation limits (65535 members and 67108863 sites) are dropped; uint32 array
 *   indexing limits remain. Unrequested scores are not computed.
 * - Non-finite ensemble members give NaN CRPS/energy; non-finite observations follow IEEE
 *   arithmetic. Variograms propagate NaN, including out-of-range pairs, as the kernel does.
 * - Subnormals and finite intermediates beyond float32 range are retained in float64. The plain
 *   numeric variogramOrder prop is evaluated in float64, rather than rounded in a shader literal.
 *
 * @throws If ensemble/observations are too short, pairs are incomplete, or weights are too short.
 */
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
  const pairDenominator = memberCount * (options.fair ? memberCount - 1 : memberCount);
  if (scores.includes('crps')) {
    const crps = new Float32Array(dimension);
    for (let site = 0; site < dimension; site++) {
      const sorted: number[] = [];
      let observationSum = 0;
      for (let member = 0; member < memberCount; member++) {
        const value = ensemble[member * dimension + site];
        sorted.push(value);
        observationSum += Math.abs(value - observations[site]);
      }
      if (!sorted.every(Number.isFinite)) {
        crps[site] = NaN;
        continue;
      }
      sorted.sort((first, second) => first - second);
      let pairSum = 0;
      for (let member = 1; member < memberCount; member++)
        pairSum += (sorted[member] - sorted[member - 1]) * member * (memberCount - member);
      crps[site] = observationSum / memberCount - pairSum / pairDenominator;
    }
    result.crps = crps;
  }
  if (scores.includes('energyScore')) {
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
      if (first >= dimension || second >= dimension) {
        total = NaN;
        break;
      }
      const observed = Math.abs(observations[first] - observations[second]) ** order;
      let forecastSum = 0;
      for (let member = 0; member < memberCount; member++)
        forecastSum +=
          Math.abs(ensemble[member * dimension + first] - ensemble[member * dimension + second]) **
          order;
      const difference = observed - forecastSum / memberCount;
      total += (variogramWeights ? variogramWeights[pair] : 1) * difference * difference;
    }
    result.variogramScore = Float32Array.of(total);
  }
  return result;
}

/** CPU twin; removes GPU invocation limits and computes only the requested scores. */
export class CPUEnsembleScoresOperation extends GPUCPUOperation {
  constructor(props: EnsembleScoresOperationProps) {
    const id = props.id ?? 'ensemble-scores';
    const scores = (
      Object.keys(props.output) as ('crps' | 'energyScore' | 'variogramScore')[]
    ).filter(score => props.output[score] !== undefined);
    const options = {...props, scores};
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
        scores.map(score => [
          props.output[score]!,
          gpuGraph.vector('float32', {length: score === 'crps' ? props.dimension : 1})
        ])
      ),
      execute({inputs}) {
        const result = computeEnsembleScoresOnCPU(
          {
            ensemble: getCPUOperationFloat32Rows(id, inputs, 'ensemble'),
            observations: getCPUOperationFloat32Rows(id, inputs, 'observations'),
            variogramPairs:
              props.variogramPairs !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'variogramPairs', 'uint32x2')
                : undefined,
            variogramWeights:
              props.variogramWeights !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'variogramWeights')
                : undefined
          },
          options
        );
        return Object.fromEntries(scores.map(score => [props.output[score]!, result[score]!]));
      }
    });
  }
}
