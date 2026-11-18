// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Refuse status-bearing weights for regression forms that cannot consume loss flags. */
export function validateCompleteRegressionWeights(id: string, weights: object): void {
  const port = weights as {status?: unknown; chunks?: readonly {weights: object}[]};
  validateRequirement(
    id,
    port.status === undefined,
    'weights.status is unsupported by named/CPU regression forms; use a GPU contributor for loss-bearing weights'
  );
  for (const chunk of port.chunks ?? []) validateCompleteRegressionWeights(id, chunk.weights);
}

/** Packed, complete CSR rows used by regression twins. @internal */
export type CPURegressionWeights = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

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
  if (weights.offsets.length < 2 || weights.offsets[0] !== 0)
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
