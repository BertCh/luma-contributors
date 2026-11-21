// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';

export type CoefficientFixture = {
  values: Float32Array;
  mask: Uint32Array;
  neighbors: number[];
  weights: number[];
  selfWeight: number;
  count: number;
  mean: number;
  deviation: number;
  sliceCount?: number;
};

export function createDenseCoefficientFixture(
  variant: 'self' | 'first' | 'relabeled',
  sign = 1
): CoefficientFixture {
  const identities = Array.from({length: 8000}, (_, row) => row);
  if (variant === 'relabeled')
    [identities[1], identities[7999]] = [identities[7999], identities[1]];
  const neighbors = Array.from({length: 7999}, (_, row) => row + 1);
  return {
    values: Float32Array.from(identities, row => Number(row >= 1600)),
    mask: new Uint32Array(8000).fill(1),
    neighbors,
    weights: neighbors.map(row =>
      variant !== 'self' && identities[row] === 1 ? 1 + sign * 2 ** -12 : 1
    ),
    selfWeight: variant === 'self' ? 1 + sign * 2 ** -12 : 1,
    count: 8000,
    mean: 0.8,
    deviation: 0.4
  };
}

export function createMaskedCoefficientFixture(relabeled: boolean): CoefficientFixture {
  const identities = Array.from({length: 257}, (_, row) => row);
  if (relabeled) [identities[1], identities[127]] = [identities[127], identities[1]];
  const neighbors = Array.from({length: 256}, (_, row) => row + 1);
  return {
    values: Float32Array.from(identities, row => (row < 16 ? -7 : row < 128 ? 1 : 0)),
    mask: Uint32Array.from(identities, row => Number(row < 128)),
    neighbors,
    weights: neighbors.map(row =>
      identities[row] >= 128 ? 0.5 : identities[row] === 1 ? 1 + 2 ** -20 : 1
    ),
    selfWeight: 1,
    count: 128,
    mean: 0,
    deviation: 2.6457513110645907
  };
}

export function createTemporalScaleFixture(scale: number): CoefficientFixture {
  return {
    values: Float32Array.of(0, 2, 1, 3, 4, NaN),
    mask: Uint32Array.of(1, 1, 1, 1, 1, 1),
    neighbors: [1, 2],
    weights: [0.5 * scale, 2 * scale],
    selfWeight: scale,
    count: 5,
    mean: 2,
    deviation: Math.sqrt(2),
    sliceCount: 2
  };
}

// Finite values are 0..4, with mean 2 and population variance 2. In cube order
// their coefficients are [1, 1, .5, .5, 2] times the common input scale. At
// unit scale the numerator is 2 and the spread is (5 * 6.5 - 5²) / 4 = 1.875.
export const TEMPORAL_SCALE_FIXTURE_SCORE = 1.0327955589886444;

export const ACCUMULATION_FIXTURE_CASES = [
  {
    name: 'preserves a 99% hot bin when rebasing a dense temporal row',
    cellCount: 16383,
    zeroCount: 3724,
    sliceCount: 2,
    selfWeight: 0.75,
    neighborWeight: 1,
    confidenceLevel: 0.99,
    hotSliceCount: 1,
    // The 16,384 observed bins include 12,659 ones. For the final window use
    // 16,382 unit coefficients and two coefficients of 3/4 on the focal zeros.
    // Gi* = (12659 - mean*sum(w)) / sqrt(variance * (n*sum(w²) - sum(w)²) / (n-1)).
    scores: [2.235570357706046, 2.607142768736541],
    probabilities: [0.025379928101670775, 0.00913012833430506]
  },
  {
    name: 'keeps a dense one-slice weighted bin nonsignificant at 95%',
    cellCount: 8192,
    zeroCount: 2048,
    sliceCount: 1,
    selfWeight: 1,
    neighborWeight: Math.fround(1 / 8191),
    confidenceLevel: 0.95,
    hotSliceCount: 0,
    // A common neighbor coefficient plus one positive excess at the focal
    // zero gives its standardized value: -0.75 / sqrt(0.75 * 0.25) = -sqrt(3).
    scores: [-1.7320508075688772],
    probabilities: [0.08326451666355043]
  }
] as const;

export function createAccumulationFixture(
  fixture: (typeof ACCUMULATION_FIXTURE_CASES)[number]
): CoefficientFixture {
  const {cellCount, zeroCount, sliceCount, selfWeight, neighborWeight} = fixture;
  const values = new Float32Array(cellCount * sliceCount).fill(NaN);
  for (let cell = 0; cell < cellCount; cell++)
    values[cell * sliceCount] = Number(cell >= zeroCount);
  if (sliceCount === 2) values[1] = 0;
  const count = cellCount + sliceCount - 1;
  const mean = (cellCount - zeroCount) / count;
  return {
    values,
    mask: new Uint32Array(values.length).fill(1),
    neighbors: Array.from({length: cellCount - 1}, (_, index) => index + 1),
    weights: new Array<number>(cellCount - 1).fill(neighborWeight),
    selfWeight,
    count,
    mean,
    deviation: Math.sqrt(mean * (1 - mean)),
    sliceCount
  };
}

export function expectFocalCoefficientReadbacks(
  readbacks: {scores: number[]; probabilities: number[]; coldCounts: number[]},
  rowCount: number,
  score = -2,
  probability = 0.0455002638963584
): void {
  for (const values of [readbacks.scores, readbacks.probabilities, readbacks.coldCounts])
    expect(values).toHaveLength(rowCount);
  expectNumericArrayClose([readbacks.scores[0]], [score], {absoluteTolerance: 1e-4});
  expectNumericArrayClose([readbacks.probabilities[0]], [probability], {absoluteTolerance: 1e-5});
  expectNumericArrayClose([readbacks.coldCounts[0]], [1]);
}
