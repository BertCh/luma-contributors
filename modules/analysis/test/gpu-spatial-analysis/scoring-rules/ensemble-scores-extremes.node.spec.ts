// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeEnsembleScoresOnCPU} from '../../../src/gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation';

it('NUM-12 preserves the two-member fair distance from the member interval', () => {
  // For two scalar members fair CRPS = fair energy = distance(y, [min(x), max(x)]).
  // Here y = -1 and the member interval starts at zero, so both scores are exactly one.
  for (const maximum of [2 ** 24, 2 ** 53]) {
    const result = computeEnsembleScoresOnCPU(
      {ensemble: Float32Array.of(0, maximum), observations: Float32Array.of(-1)},
      {memberCount: 2, dimension: 1, estimator: 'fair'}
    );
    expect(Number.isFinite(result.crps![0])).toBe(true);
    expect(Array.from(result.crps!)).toEqual([1]);
    expect(Number.isFinite(result.energyScore![0])).toBe(true);
    expect(Array.from(result.energyScore!)).toEqual([1]);
  }
});

it('NUM-12 shares estimator defaults and rejects ambiguous or unknown estimators', () => {
  const input = {ensemble: Float32Array.of(0, 2 ** 53), observations: Float32Array.of(-1)};
  const options = {memberCount: 2, dimension: 1};
  expect(computeEnsembleScoresOnCPU(input, {...options, estimator: 'fair'}).crps![0]).toBe(1);
  expect(() =>
    computeEnsembleScoresOnCPU(input, {...options, estimator: 'fair', ...{fair: false}})
  ).toThrow(/fair was removed.*estimator/);
  expect(() =>
    computeEnsembleScoresOnCPU(input, {...options, estimator: 'other' as 'fair'})
  ).toThrow(/estimator must be fair or empirical/);
});

for (const weight of [2, 0, -1, Infinity, -Infinity, NaN]) {
  it(`NUM-07 validates variogram weight ${weight} without discarding other scores`, () => {
    // One member [0,1], observation [0,0]: discrepancy is 1, energy is 1, CRPS is [0,1].
    const result = computeEnsembleScoresOnCPU(
      {
        ensemble: Float32Array.of(0, 1),
        observations: Float32Array.of(0, 0),
        variogramPairs: Uint32Array.of(0, 1),
        variogramWeights: Float32Array.of(weight)
      },
      {memberCount: 1, dimension: 2}
    );
    expect(Array.from(result.crps!)).toEqual([0, 1]);
    expect(Array.from(result.energyScore!)).toEqual([1]);
    if (Number.isFinite(weight) && weight >= 0) {
      expect(Number.isFinite(result.variogramScore![0])).toBe(true);
      expect(result.variogramScore![0]).toBe(weight);
    } else expect(result.variogramScore![0]).toBeNaN();
  });
}

for (const endpoint of ['member-first', 'member-second', 'observed-first', 'observed-second']) {
  it(`B3 rejects nonfinite variogram ${endpoint} while retaining independent scores`, () => {
    const ensemble = Float32Array.of(1, 1);
    const observations = Float32Array.of(1, 1);
    const values = endpoint.startsWith('member') ? ensemble : observations;
    values[endpoint.endsWith('first') ? 0 : 1] = Infinity;
    const result = computeEnsembleScoresOnCPU(
      {ensemble, observations, variogramPairs: Uint32Array.of(0, 1)},
      {memberCount: 1, dimension: 2}
    );
    expect(Array.from(result.variogramScore!)).toEqual([NaN]);
    // Invalid member rows make their CRPS and the whole-field energy unavailable; other CRPS remains zero.
    const expectedCrps = endpoint.endsWith('first')
      ? [endpoint.startsWith('member') ? NaN : Infinity, 0]
      : [0, endpoint.startsWith('member') ? NaN : Infinity];
    expect(Array.from(result.crps!)).toEqual(expectedCrps);
    expect(Array.from(result.energyScore!)).toEqual([
      endpoint.startsWith('member') ? NaN : Infinity
    ]);
  });
}

it('S4 keeps a perfect high-order CPU variogram zero without forming overflowing powers', () => {
  const values = Float32Array.of(4096, 0);
  const result = computeEnsembleScoresOnCPU(
    {ensemble: values, observations: values, variogramPairs: Uint32Array.of(0, 1)},
    {memberCount: 1, dimension: 2, variogramOrder: 100}
  );
  expect(Array.from(result.crps!)).toEqual([0, 0]);
  expect(Array.from(result.energyScore!)).toEqual([0]);
  expect(Array.from(result.variogramScore!)).toEqual([0]);
});
