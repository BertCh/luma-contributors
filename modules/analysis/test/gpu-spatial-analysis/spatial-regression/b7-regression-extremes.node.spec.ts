// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialWeightsTracesOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation';
import {
  getStudentTSurvival,
  getStudentTTwoSidedPValue
} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';

it('R14 rounds the two-sided Student t tail once at the binary64 floor', () => {
  // /usr/bin/python3, mpmath at 65 digits: betainc(df/2, .5, 0, df/(df+t*t),
  // regularized=True) gives 1.93085795121664e-323, 6.08496407875726e-324 and
  // 1.91591002891675e-324, respectively. Round each to binary64.
  expect(getStudentTTwoSidedPValue(38.45, 1e10)).toBe(4 * Number.MIN_VALUE);
  expect(getStudentTTwoSidedPValue(38.48, 1e10)).toBe(Number.MIN_VALUE);
  expect(getStudentTTwoSidedPValue(-38.48, 1e10)).toBe(Number.MIN_VALUE);
  expect(getStudentTTwoSidedPValue(38.51, 1e10)).toBe(0);
});

it('R14 retains the ordinary-magnitude large-df Student t tail', () => {
  // /usr/bin/python3: 2 * scipy.stats.t.sf(2, 10**10).
  expect(getStudentTTwoSidedPValue(2, 1e10)).toBeCloseTo(0.04550026392335381, 14);
  expect(getStudentTSurvival(2, 1e10)).toBeCloseTo(0.022750131961676905, 14);
});

it('R14 rounds the one-sided Student t tail once at the binary64 floor', () => {
  // /usr/bin/python3 Decimal, 70 digits: a 12-term integration-by-parts expansion of
  // the Student t density gives 2.50949614320884e-324 and 2.46163655254249e-324.
  // The magnitude of the next term bounds each integration remainder below 8e-351.
  expect(getStudentTSurvival(38.485, 1e11)).toBe(Number.MIN_VALUE);
  expect(getStudentTSurvival(38.4855, 1e11)).toBe(0);
  // The same independent expansion at df=1e9 gives 2.51086082433138e-324
  // and 2.46297527683408e-324 on the two sides of the rounding boundary.
  expect(getStudentTSurvival(38.485, 1e9)).toBe(Number.MIN_VALUE);
  expect(getStudentTSurvival(38.4855, 1e9)).toBe(0);
});

it.each([
  [2, 2],
  [3.2e38, 3.19999997882106e38]
])('N1 CPU traces retain scale and bounded powers for edge weight %s', (weight, norm) => {
  // W / s is the two-row swap matrix, whose even powers are the identity.
  // Python NumPy float64 matrix powers after converting the weights from float32 give the pins.
  const result = computeSpatialWeightsTracesOnCPU(
    {
      weights: {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(1, 0),
        weights: Float32Array.of(weight, weight)
      }
    },
    {maximumPower: 4, mode: 'exact'}
  );
  expect(Array.from(result.infinityNorm)).toEqual([norm]);
  expect(Array.from(result.powerTraces)).toEqual([0, 2, 0, 2]);
  expect(Array.from(result.powerSums)).toEqual([2, 2, 2, 2]);
});

it.each(['exact', 'stochastic'] as const)('B1 CPU retains the isolated cube in %s mode', mode => {
  const result = computeSpatialWeightsTracesOnCPU(
    {
      weights: {
        offsets: Uint32Array.of(0, 1, 1, 2),
        neighbors: Uint32Array.of(1, 2),
        weights: Float32Array.of(1.875 * 2 ** 127, 2)
      }
    },
    {maximumPower: 4, mode, probeCount: 3, includeGramTraces: true}
  );
  // W'W is diagonal. Its product with W has just one nonzero diagonal, 2² × 2.
  expect(Array.from(result.gramTraces!)).toEqual([Infinity, 8, Infinity, Infinity]);
});
