// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialWeightsCSR} from './spatial-weights-traces-parameters';

/** Exact structural certificate W=cI; rounded equal eigenvalues alone are insufficient. @internal */
export function hasSpatialErrorScalarWeights(
  weights: SpatialWeightsCSR,
  rowCount: number,
  scalarWeight: number
): boolean {
  if (!Number.isFinite(scalarWeight)) return false;
  for (let row = 0; row < rowCount; row++) {
    let diagonalWeight = 0;
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      const neighbor = weights.neighbors[slot];
      if (neighbor >= rowCount) continue;
      if (neighbor === row) diagonalWeight = weights.weights[slot];
      else if (weights.weights[slot] !== 0) return false;
    }
    if (diagonalWeight !== scalarWeight) return false;
  }
  return true;
}

/** One invocation per row; wrap with the existing chunk row adapter when necessary. @internal */
export function getSpatialErrorScalarCertificateWGSL(): string {
  return `let scalarWeight = eigenvalues[eigenvaluesOffset];
  var scalarRow = isFiniteValue(scalarWeight) && eigenvalues[eigenvaluesOffset + index] == scalarWeight;
  var diagonalWeight = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) { continue; }
    let weight = weights[weightsOffset + slot];
    if (neighbor == index) {
      diagonalWeight = weight;
    } else if ((bitcast<u32>(weight) & 0x7fffffffu) != 0u) {
      // Inspect stored bits so a subnormal link cannot be certified as zero by float flushing.
      scalarRow = false;
    }
  }
  scalarDifferences[scalarDifferencesOffset + index] = select(1.0, 0.0, scalarRow && diagonalWeight == scalarWeight);`;
}
