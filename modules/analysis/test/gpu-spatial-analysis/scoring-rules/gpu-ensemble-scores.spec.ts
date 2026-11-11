// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {runEnsembleScores, runGaussianScores, SCORING_FIXTURE} from './scoring-rules-harness';

// Oracle: test/oracles/scoring-rules.py (binary64 evaluations of the scoringRules 1.1.3
// definitions, cross-checked by integration; the scoringRules R output itself is pending).
const {tolerances} = SCORING_FIXTURE;

function expectRelative(actual: number, expected: number, relative: number, label: string) {
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    relative * Math.max(1, Math.abs(expected))
  );
}

it('GPUEnsembleScores sample CRPS, energy and variogram scores match the references', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const entry of SCORING_FIXTURE.ensembles) {
    const nonFair = await runEnsembleScores(device, entry, {
      crps: true,
      energyScore: true,
      variogramPairs: entry.variogramPairsAll,
      variogramOrder: entry.variogramOrder
    });
    entry.crps.forEach((expected, site) =>
      expectRelative(
        nonFair.crps![site],
        expected,
        tolerances.crpsRelative,
        `${entry.name} crps ${site}`
      )
    );
    expectRelative(
      nonFair.energyScore!,
      entry.energyScore,
      tolerances.energyRelative,
      `${entry.name} energy`
    );
    expectRelative(
      nonFair.variogramScore!,
      entry.variogramScoreAll,
      tolerances.variogramRelative,
      `${entry.name} variogram`
    );
    const fair = await runEnsembleScores(device, entry, {
      fair: true,
      crps: true,
      energyScore: true
    });
    entry.crpsFair.forEach((expected, site) =>
      expectRelative(
        fair.crps![site],
        expected,
        tolerances.crpsRelative,
        `${entry.name} fair crps ${site}`
      )
    );
    expectRelative(
      fair.energyScore!,
      entry.energyScoreFair,
      tolerances.energyRelative,
      `${entry.name} fair energy`
    );
    // Teeth: the fair and non-fair forms differ by the documented pair-term factor.
    expect(Math.abs(fair.energyScore! - nonFair.energyScore!)).toBeGreaterThan(1e-4);
    // Neighbour pairs with weights, at order 0.5 and order 1.
    for (const [order, expected] of [
      [0.5, entry.variogramScoreNeighbours],
      [1, entry.variogramScoreNeighboursOrderOne]
    ] as const) {
      const restricted = await runEnsembleScores(device, entry, {
        variogramPairs: entry.variogramPairsNeighbours,
        variogramWeights: entry.variogramWeights,
        variogramOrder: order
      });
      expectRelative(
        restricted.variogramScore!,
        expected,
        tolerances.variogramRelative,
        `${entry.name} variogram order ${order}`
      );
    }
  }
}, 120000);

it('GPUEnsembleScores reproduces the hand-computed CRPS (5/9, fair 1/3)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {hand} = SCORING_FIXTURE;
  const entry = {
    memberCount: hand.members.length,
    dimension: 1,
    ensemble: hand.members,
    observations: [hand.observation]
  };
  expect((await runEnsembleScores(device, entry, {crps: true})).crps![0]).toBeCloseTo(5 / 9, 6);
  expect((await runEnsembleScores(device, entry, {crps: true, fair: true})).crps![0]).toBeCloseTo(
    1 / 3,
    6
  );
});

it('GPUEnsembleScores is bitwise repeatable and independent of workgroup scheduling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const entry = SCORING_FIXTURE.ensembles.find(item => item.name === 'members-over-lanes')!;
  const options = {crps: true, energyScore: true, variogramPairs: entry.variogramPairsAll};
  const [first, second] = [
    await runEnsembleScores(device, entry, options),
    await runEnsembleScores(device, entry, options)
  ];
  expect(Array.from(second.crps!)).toEqual(Array.from(first.crps!));
  expect(second.energyScore).toBe(first.energyScore);
  expect(second.variogramScore).toBe(first.variogramScore);
});

it('GPUGaussianPredictiveScores matches the closed-form Gaussian CRPS', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {gaussian} = SCORING_FIXTURE;
  const result = await runGaussianScores(
    device,
    gaussian.map(entry => entry.mean),
    gaussian.map(entry => entry.variance),
    gaussian.map(entry => entry.observation)
  );
  gaussian.forEach((entry, index) => {
    expect(
      Math.abs(result[index] - entry.crps),
      `mean ${entry.mean} variance ${entry.variance} y ${entry.observation}: ${result[index]} vs ${entry.crps}`
    ).toBeLessThanOrEqual(
      tolerances.gaussianAbsolute + tolerances.gaussianRelative * Math.abs(entry.crps)
    );
  });
  // Teeth: the cases span the centre, the tails (|z| beyond the erf series cut at 5.66) and a point forecast.
  expect(
    Math.max(
      ...gaussian.map(
        entry => Math.abs(entry.observation - entry.mean) / Math.sqrt(entry.variance || 1)
      )
    )
  ).toBeGreaterThan(5);
  // Negative and NaN variances score NaN.
  const invalid = await runGaussianScores(device, [0, 0], [-1, NaN], [0, 0]);
  expect(Number.isNaN(invalid[0]) && Number.isNaN(invalid[1])).toBe(true);
}, 60000);

it('a sample CRPS ensemble converges to the Gaussian CRPS', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Quantile ensemble of N(0, 1): the sample CRPS at y = 1 approaches crps_norm(1; 0, 1) = 0.6024...
  const memberCount = 200;
  const inverseNormal = (probability: number) => {
    // Bisection on the CDF.
    let [lower, upper] = [-6, 6];
    for (let step = 0; step < 80; step++) {
      const middle = 0.5 * (lower + upper);
      const cdf = 0.5 * (1 + erf(middle / Math.SQRT2));
      if (cdf < probability) lower = middle;
      else upper = middle;
    }
    return 0.5 * (lower + upper);
  };
  const members = Array.from({length: memberCount}, (_, index) =>
    inverseNormal((index + 0.5) / memberCount)
  );
  const result = await runEnsembleScores(
    device,
    {memberCount, dimension: 1, ensemble: members, observations: [1]},
    {crps: true}
  );
  const closed = SCORING_FIXTURE.gaussian.find(
    entry => entry.observation === 1 && entry.mean === 0 && entry.variance === 1
  )!;
  expect(Math.abs(result.crps![0] - closed.crps)).toBeLessThan(2e-3);
});

it('GPUEnsembleScores variogram score is NaN for a NaN observation or member, and 0^p is 0', async () => {
  // Math correctness review, interpolation-density-04: `|difference|^p` was guarded by
  // `magnitude > 0`, which is false for NaN, so a NaN input scored as a zero difference.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const options = {variogramPairs: [[0, 1]], variogramOrder: 0.5};
  // The report's case: with the NaN read as 0 the score was (0 - |1 - 2|^0.5)^2 = 1.
  const nanObservation = await runEnsembleScores(
    device,
    {memberCount: 1, dimension: 2, ensemble: [1, 2], observations: [Number.NaN, 2]},
    options
  );
  expect(Number.isNaN(nanObservation.variogramScore)).toBe(true);
  // With the NaN read as 0 the score was (|1 - 3|^0.5 - 1 / 2)^2 = 0.8358.
  const nanMember = await runEnsembleScores(
    device,
    {memberCount: 2, dimension: 2, ensemble: [1, 2, Number.NaN, 5], observations: [1, 3]},
    options
  );
  expect(Number.isNaN(nanMember.variogramScore)).toBe(true);
  const infiniteMember = await runEnsembleScores(
    device,
    {memberCount: 1, dimension: 2, ensemble: [Number.POSITIVE_INFINITY, 5], observations: [1, 3]},
    options
  );
  expect(Number.isFinite(infiniteMember.variogramScore)).toBe(false);
  // Equal observations: |2 - 2|^0.5 = 0 (not NaN), forecast |1 - 3|^0.5 = sqrt(2), score 2.
  const equalObservations = await runEnsembleScores(
    device,
    {memberCount: 1, dimension: 2, ensemble: [1, 3], observations: [2, 2]},
    options
  );
  expect(equalObservations.variogramScore).toBeCloseTo(2, 5);
});

/** High-precision erf by the Taylor series (the test's own, independent of the GPU kernel). */
function erf(x: number): number {
  let term = x;
  let sum = x;
  for (let n = 1; n < 200; n++) {
    term *= (-x * x) / n;
    sum += term / (2 * n + 1);
  }
  return (2 / Math.sqrt(Math.PI)) * sum;
}
