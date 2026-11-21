// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import fairFixture from '../../oracles/fixtures/scoring-rules-fair.json';
import {runEnsembleScores} from './scoring-rules-harness';

// Oracle: test/oracles/scoring-rules-fair.py (NumPy 2.0.2, SciPy 1.13.1). Math correctness review
// 2026-11-10, interpolation-density-12: the fair CRPS and the fair energy score had no reference
// outside the pairwise sum the kernel itself uses. The fixture gives two that do not share it:
// another formula per ensemble (order statistics from `numpy.sort`; the mean over distinct pairs
// from `scipy.spatial.distance.pdist`), and the property that defines "fair": the mean over many
// ensembles drawn from a distribution is the score of that distribution, here a closed form from
// `scipy.stats.norm` and `scipy.special.hyp1f1`. The standard estimators are biased high by
// more than the band, so the second check tells the divisor `2 m (m - 1)` from `2 m^2`.

const SCORE_TOLERANCE = 2e-5;

function getMean(values: ArrayLike<number>): number {
  let sum = 0;
  for (let index = 0; index < values.length; index++) {
    sum += values[index];
  }
  return sum / values.length;
}

it('GPUEnsembleScores fair CRPS matches the order-statistic form and is unbiased for the Gaussian CRPS', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {crps} = fairFixture;
  const entry = {
    memberCount: crps.memberCount,
    dimension: crps.siteCount,
    ensemble: crps.ensemble,
    observations: new Array<number>(crps.siteCount).fill(crps.observation)
  };
  const fair = (await runEnsembleScores(device, entry, {crps: true, estimator: 'fair'})).crps!;
  const standard = (await runEnsembleScores(device, entry, {crps: true})).crps!;
  expect(fair.length).toBe(crps.siteCount);
  for (let site = 0; site < crps.siteCount; site++) {
    expect(Math.abs(fair[site] - crps.fair[site]), `fair site ${site}`).toBeLessThanOrEqual(
      SCORE_TOLERANCE * Math.max(1, Math.abs(crps.fair[site]))
    );
    expect(
      Math.abs(standard[site] - crps.standard[site]),
      `standard site ${site}`
    ).toBeLessThanOrEqual(SCORE_TOLERANCE * Math.max(1, Math.abs(crps.standard[site])));
  }
  // Fairness: the mean over 512 ensembles of five members is the CRPS of N(0, 1) itself, within
  // three standard errors. The standard estimator is high by E|X - X'| / (2 m) = 0.113.
  expect(Math.abs(getMean(fair) - crps.closedForm)).toBeLessThan(crps.meanBand);
  expect(getMean(standard) - crps.closedForm).toBeGreaterThan(2 * crps.meanBand);
});

it('GPUEnsembleScores fair energy score matches the distinct-pair form and is unbiased for the Gaussian energy score', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {energy} = fairFixture;
  const fair: number[] = [];
  const standard: number[] = [];
  for (const [index, ensemble] of energy.ensembles.entries()) {
    const entry = {
      memberCount: energy.memberCount,
      dimension: energy.dimension,
      ensemble,
      observations: energy.observation
    };
    const fairScore = (
      await runEnsembleScores(device, entry, {energyScore: true, estimator: 'fair'})
    ).energyScore!;
    const standardScore = (await runEnsembleScores(device, entry, {energyScore: true}))
      .energyScore!;
    expect(Math.abs(fairScore - energy.fair[index]), `fair ensemble ${index}`).toBeLessThanOrEqual(
      SCORE_TOLERANCE * Math.max(1, Math.abs(energy.fair[index]))
    );
    expect(
      Math.abs(standardScore - energy.standard[index]),
      `standard ensemble ${index}`
    ).toBeLessThanOrEqual(SCORE_TOLERANCE * Math.max(1, Math.abs(energy.standard[index])));
    fair.push(fairScore);
    standard.push(standardScore);
  }
  // Fairness: the mean over 128 ensembles of three members is the energy score of N(0, I_3)
  // itself. The standard estimator is high by E||X - X'|| / (2 m) = 0.376.
  expect(Math.abs(getMean(fair) - energy.closedForm)).toBeLessThan(energy.meanBand);
  expect(getMean(standard) - energy.closedForm).toBeGreaterThan(1.5 * energy.meanBand);
}, 240000);
