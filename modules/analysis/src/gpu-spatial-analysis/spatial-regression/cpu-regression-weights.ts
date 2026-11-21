// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  hasCPUWeightsLoss,
  type CPUWeightsLossInput,
  type CPUWeightsStatus
} from '../spatial-weights/cpu-spatial-weights-loss';

/** Accept concrete clean CPU flags; refuse unknown or incomplete certificates. */
export function validateCompleteRegressionWeights(id: string, weights: object): void {
  const named = weights as {
    offsets?: unknown;
    status?: unknown;
    chunks?: readonly {weights: object}[];
  };
  if (
    typeof named.offsets === 'string' ||
    named.chunks?.some(chunk => typeof (chunk.weights as {offsets?: unknown}).offsets === 'string')
  ) {
    validateRequirement(
      id,
      named.status === undefined,
      'weights.status is unsupported by named/CPU regression forms; use a GPU contributor for loss-bearing weights'
    );
    for (const name of ['count', 'requiredCount', 'overflow', 'candidateOverflow'] as const) {
      const value = (weights as Record<string, unknown>)[name];
      if (value !== undefined)
        validateRequirement(
          id,
          typeof value === 'string' && value.length > 0,
          `${name} must name a non-empty graph value; received ${String(value)}`
        );
    }
    for (const chunk of named.chunks ?? []) validateCompleteRegressionWeights(id, chunk.weights);
    return;
  }
  const port = weights as CPUWeightsLossInput;
  validateRequirement(
    id,
    !hasCPUWeightsLoss(id, port),
    'weights.status indicates incomplete weights; rebuild the producer with sufficient candidate and final capacity or use a GPU contributor for loss-bearing weights'
  );
  for (const chunk of named.chunks ?? []) validateCompleteRegressionWeights(id, chunk.weights);
}

/** Packed, complete CSR rows used by regression twins. @internal */
export type CPURegressionWeights = CPUWeightsLossInput & {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

/** Scalar metadata attached to an uploaded CPU CSR. */
export type CPURegressionWeightsStatus = CPUWeightsStatus;

/** Flags and retained CSR extent determine completeness. @internal */
export function hasCPURegressionWeightsLoss(
  id: string,
  weights: CPURegressionWeights & CPUWeightsLossInput
): boolean {
  return hasCPUWeightsLoss(id, weights);
}

/**
 * Validates CSR array consistency, retaining unused capacity and invalid neighbor sentinels.
 * Status-bearing inputs are refused unless the caller explicitly handles loss before publishing.
 * @internal
 */
export function validateCPURegressionWeights(
  id: string,
  weights: CPURegressionWeights,
  statusPolicy: 'reject' | 'handled-by-caller' = 'reject'
): number {
  if (statusPolicy === 'reject') validateCompleteRegressionWeights(id, weights);
  if (weights.offsets.length < 1 || weights.offsets[0] !== 0)
    throw new Error(`${id} weights.offsets must start at zero and hold rowCount + 1 entries`);
  if (weights.weights.length !== weights.neighbors.length)
    throw new Error(`${id} weights.weights length must equal weights.neighbors length`);
  let previous = 0;
  for (const offset of weights.offsets) {
    if (offset < previous || offset > weights.neighbors.length)
      throw new Error(
        `${id} weights.offsets must be nondecreasing and within weights.neighbors capacity`
      );
    previous = offset;
  }
  return weights.offsets.length - 1;
}
