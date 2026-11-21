// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runEnsembleScores} from './scoring-rules-harness';

it('NUM-12 preserves fair scalar interval distances on every output selection', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // With two members, the fair score is distance(y, [min(x), max(x)]), exactly 1 here.
  for (const maximum of [2 ** 24, 2 ** 53]) {
    for (const selection of [{crps: true}, {energyScore: true}, {crps: true, energyScore: true}]) {
      const actual = await runEnsembleScores(
        device,
        {
          memberCount: 2,
          dimension: 1,
          ensemble: [0, maximum],
          observations: [-1]
        },
        {...selection, estimator: 'fair'}
      );
      if (selection.crps) {
        expect(Number.isFinite(actual.crps![0])).toBe(true);
        expect(actual.crps![0]).toBe(1);
      }
      if (selection.energyScore) {
        expect(Number.isFinite(actual.energyScore)).toBe(true);
        expect(actual.energyScore).toBe(1);
      }
    }
  }
  // A tied interval contributes zero; the sole nonzero fair coefficient is 2*1/(3*2).
  const tied = await runEnsembleScores(
    device,
    {
      memberCount: 3,
      dimension: 1,
      ensemble: [0, 0, 2],
      observations: [1]
    },
    {crps: true, energyScore: true, estimator: 'fair'}
  );
  expect(Number.isFinite(tied.crps![0])).toBe(true);
  expect(tied.crps![0]).toBeCloseTo(1 / 3, 6);
  expect(Number.isFinite(tied.energyScore)).toBe(true);
  expect(tied.energyScore).toBeCloseTo(1 / 3, 6);
});

it('WGS-09 returns finite zero for an extreme perfect variogram forecast', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = [-(2 ** 127), 2 ** 127];
  const actual = await runEnsembleScores(
    device,
    {
      memberCount: 1,
      dimension: 2,
      ensemble: values,
      observations: values
    },
    {variogramPairs: [[0, 1]], variogramOrder: 0.5}
  );
  expect(Number.isFinite(actual.variogramScore)).toBe(true);
  expect(actual.variogramScore).toBe(0);
  const fractional = await runEnsembleScores(
    device,
    {
      memberCount: 1,
      dimension: 2,
      ensemble: values,
      observations: [0, 0]
    },
    {variogramPairs: [[0, 1]], variogramOrder: 0.497}
  );
  // Python f32(2**(256*f32(.497))); the squared powered gap is representable.
  const offset = await runEnsembleScores(
    device,
    {
      memberCount: 1,
      dimension: 2,
      ensemble: [2 ** 29, 2 ** 29 + 64],
      observations: [0, 0]
    },
    {variogramPairs: [[0, 1]], variogramOrder: 10}
  );
  expect(Number.isFinite(offset.variogramScore)).toBe(true);
  expect(Math.abs(offset.variogramScore! / 1.329227995784916e36 - 1)).toBeLessThan(3e-5);
  expect(Number.isFinite(fractional.variogramScore)).toBe(true);
  expect(Math.abs(fractional.variogramScore! / 1.998246724570999e38 - 1)).toBeLessThan(3e-5);
});

it('NUM-07 invalid weights affect only the requested variogram score', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One member [0,1] versus [0,0]: energy=1, CRPS=[0,1], squared pair discrepancy=1.
  for (const weight of [2, 0, -0, -(2 ** -149), -1, Infinity, -Infinity, NaN]) {
    const actual = await runEnsembleScores(
      device,
      {
        memberCount: 1,
        dimension: 2,
        ensemble: [0, 1],
        observations: [0, 0]
      },
      {crps: true, energyScore: true, variogramPairs: [[0, 1]], variogramWeights: [weight]}
    );
    expect(Array.from(actual.crps!)).toEqual([0, 1]);
    expect(actual.energyScore).toBe(1);
    if (Number.isFinite(weight) && weight >= 0) {
      expect(Number.isFinite(actual.variogramScore)).toBe(true);
      expect(actual.variogramScore).toBe(weight === 0 ? 0 : weight);
    } else expect(actual.variogramScore).toBeNaN();
  }
});

it('R1 variogram finite accuracy and exponent recovery', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const constant = Array.from({length: 20000}, (_, index) => index % 2);
  const cases = [
    // Two identical gaps, so (10,000,001 - 10,000,000)^2 = 1.
    {ensemble: [0, 10000000, 0, 10000000], observations: [0, 10000001], order: 1, expected: 1},
    // All 10,000 members have powered gap one, so their mean and squared discrepancy are one.
    {ensemble: constant, observations: [0, 0], order: 0.5, expected: 1},
    // 2^(-149/2), 2^(-126/2); fractional powers recover normal outputs from tiny gaps.
    {ensemble: [0, 2 ** -149], observations: [0, 0], order: 0.25, expected: 3.743392066509216e-23},
    {ensemble: [0, 2 ** -126], observations: [0, 0], order: 0.25, expected: 1.0842021724855044e-19},
    // Perfect forecast, even though the individual 100th powers overflow binary64.
    {ensemble: [4096, 0], observations: [4096, 0], order: 100, expected: 0}
  ];
  for (const entry of cases) {
    const result = await runEnsembleScores(
      device,
      {
        memberCount: entry.ensemble.length / 2,
        dimension: 2,
        ensemble: entry.ensemble,
        observations: entry.observations
      },
      {variogramPairs: [[0, 1]], variogramOrder: entry.order}
    );
    expect(Number.isFinite(result.variogramScore)).toBe(true);
    if (entry.expected === 0) expect(result.variogramScore).toBe(0);
    else expect(Math.abs(result.variogramScore! / entry.expected - 1)).toBeLessThan(3e-5);
  }
});

it('R1 nonfinite variogram endpoints preserve independently requested scores', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const estimator of ['empirical', 'fair'] as const) {
    const memberCount = estimator === 'fair' ? 2 : 1;
    for (const endpoint of ['member-first', 'member-second', 'observed-first', 'observed-second']) {
      const ensemble = Array.from({length: memberCount * 2}, () => 1);
      const observations = [1, 1];
      const values = endpoint.startsWith('member') ? ensemble : observations;
      values[endpoint.endsWith('first') ? 0 : 1] = Infinity;
      const result = await runEnsembleScores(
        device,
        {ensemble, observations, memberCount, dimension: 2},
        {estimator, crps: true, energyScore: true, variogramPairs: [[0, 1]]}
      );
      expect(result.variogramScore).toBeNaN();
      const affected = endpoint.startsWith('member') ? NaN : Infinity;
      expect(Array.from(result.crps!)).toEqual(
        endpoint.endsWith('first') ? [affected, 0] : [0, affected]
      );
      if (endpoint.startsWith('member')) expect(result.energyScore).toBeNaN();
      else expect(result.energyScore).toBe(Infinity);
    }
  }
});
