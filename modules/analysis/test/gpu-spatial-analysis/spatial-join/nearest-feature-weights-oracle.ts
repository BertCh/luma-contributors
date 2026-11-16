// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OracleCSR} from '../spatial-weights/spatial-weights-oracle';

/** Cross weights have query rows and feature columns in separate ID spaces. */
export function assertValidCrossWeights(
  weights: OracleCSR,
  queryCount: number,
  featureCount: number,
  label: string
): void {
  if (weights.offsets.length !== queryCount + 1 || weights.offsets[0] !== 0) {
    throw new Error(`${label}: offsets must hold queryCount + 1 entries starting at 0`);
  }
  const slotCount = weights.neighbors.length;
  if (
    weights.offsets[queryCount] !== slotCount ||
    weights.weights.length !== slotCount ||
    weights.distances.length !== slotCount
  ) {
    throw new Error(`${label}: offsets and output columns must agree on slot count`);
  }
  for (let row = 0; row < queryCount; row++) {
    const begin = weights.offsets[row];
    const end = weights.offsets[row + 1];
    if (!Number.isInteger(end) || end < begin || end > slotCount) {
      throw new Error(`${label}: invalid offsets at query row ${row}`);
    }
    for (let slot = begin; slot < end; slot++) {
      const neighbor = weights.neighbors[slot];
      if (!Number.isInteger(neighbor) || neighbor < 0 || neighbor >= featureCount) {
        throw new Error(`${label}: neighbor out of feature range`);
      }
      if (slot > begin && neighbor <= weights.neighbors[slot - 1]) {
        throw new Error(`${label}: query row ${row} neighbors are not strictly ascending`);
      }
      if (!(weights.weights[slot] >= 0) || !Number.isFinite(weights.weights[slot])) {
        throw new Error(`${label}: weight at slot ${slot} is not finite and non-negative`);
      }
      if (!(weights.distances[slot] >= 0) || !Number.isFinite(weights.distances[slot])) {
        throw new Error(`${label}: distance at slot ${slot} is not finite and non-negative`);
      }
    }
  }
}
