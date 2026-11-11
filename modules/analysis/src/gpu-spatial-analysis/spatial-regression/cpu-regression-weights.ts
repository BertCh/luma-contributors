// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Packed CSR rows used by regression twins. @internal */
export type CPURegressionWeights = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

/** Validates CSR array consistency, retaining unused capacity and invalid neighbor sentinels. @internal */
export function validateCPURegressionWeights(id: string, weights: CPURegressionWeights): number {
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
