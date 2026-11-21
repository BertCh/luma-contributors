// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {runHawkesGPU, runHawkesGPUSequence} from './stm08-hawkes-harness';

const WINDOW = {area: 2, duration: 10, maximumLag: 10, maximumDistance: 10, maximumParents: 1};
const INITIAL = {background: 0.1, branchingRatio: 0.5, decayRate: 1, spatialStandardDeviation: 1};

it('retains the one earlier parent for every member of a simultaneous group', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const events = [{x: 0, y: 0, time: 1}, ...Array.from({length: 3}, () => ({x: 1, y: 0, time: 2}))];
  const result = await runHawkesGPU(device, events, WINDOW, INITIAL, {
    maximumIterations: 1,
    tolerance: 1e-12
  });
  // Every time-2 event has exactly the same one parent. Compute the E/M step independently.
  const intensityTerm = (0.5 / (2 * Math.PI)) * Math.exp(-1.5);
  const responsibility = intensityTerm / (0.1 + intensityTerm);
  expect(result.estimates.background).toBeCloseTo((4 - 3 * responsibility) / 20, 6);
  // Independent SciPy score root with remaining times [9,8,8,8], S=3*r, L=S.
  expect(result.estimates.branchingRatio).toBeCloseTo(0.11312306432164705, 6);
  expect(result.estimates.decayRate).toBeCloseTo(0.9976705756605463, 6);
  expect(result.estimates.spatialStandardDeviation).toBeCloseTo(Math.SQRT1_2, 6);
  expect(result.candidateOverflow).toBe(0);
  expect(result.backgroundProbability[1]).toBeCloseTo(result.backgroundProbability[2], 6);
  expect(result.backgroundProbability[1]).toBeCloseTo(result.backgroundProbability[3], 6);
});

it('reports no parent truncation for a catalogue containing only simultaneous events', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const events = Array.from({length: 20}, () => ({x: 0, y: 0, time: 2}));
  const result = await runHawkesGPU(device, events, WINDOW, INITIAL, {maximumIterations: 1});
  expect(result.candidateOverflow).toBe(0);
  expect(result.estimates.background).toBeCloseTo(1, 6);
  expect(result.backgroundProbability).toEqual(new Array(20).fill(1));
});

it.each<Readonly<Record<number, number>>>([
  {4: Infinity},
  {8: 1e30},
  {8: 1e-30}
])('does not start EM with invalid runtime parameters %j', async parameterOverride => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runHawkesGPU(
    device,
    [
      {x: 0, y: 0, time: 1},
      {x: 1, y: 0, time: 2}
    ],
    WINDOW,
    INITIAL,
    {maximumIterations: 1, parameterOverride}
  );
  expect(result.iterationCount).toBe(0);
  expect(Object.values(result.estimates).every(Number.isNaN)).toBe(true);
  expect(result.backgroundProbability.every(Number.isNaN)).toBe(true);
});

it('clears every diagnostic when valid parameters become invalid on a compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const events = [
    {x: 0, y: 0, time: 0},
    {x: 1, y: 0, time: 1},
    {x: 0, y: 0, time: 2},
    {x: 1, y: 0, time: 3}
  ];
  const [valid, invalid, recovered] = await runHawkesGPUSequence(
    device,
    events,
    WINDOW,
    INITIAL,
    {maximumIterations: 1},
    [{}, {4: Infinity}, {}]
  );
  expect(valid.iterationCount).toBe(1);
  expect(valid.candidateOverflow).toBeGreaterThan(0);
  expect(invalid.iterationCount).toBe(0);
  expect(invalid.converged).toBe(0);
  expect(invalid.iterationLimitReached).toBe(0);
  expect(invalid.candidateOverflow).toBe(0);
  expect(invalid.invalidCount).toBe(events.length);
  expect(invalid.approximation).toBe(1);
  expect(invalid.logLikelihood).toBeNaN();
  expect(Object.values(invalid.estimates).every(Number.isNaN)).toBe(true);
  expect(invalid.backgroundProbability.every(Number.isNaN)).toBe(true);
  expect(recovered).toEqual(valid);
});

it('counts real parent overflow through timestamp ties and includes the maximum-lag boundary', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const window = {...WINDOW, maximumLag: 2, maximumParents: 2};
  const events = [
    {x: 0, y: 0, time: 0},
    {x: 1, y: 0, time: 1},
    {x: 2, y: 0, time: 1},
    ...Array.from({length: 3}, () => ({x: 3, y: 0, time: 2}))
  ];
  const capped = await runHawkesGPU(device, events, window, INITIAL, {maximumIterations: 1});
  const full = await runHawkesGPU(device, events, {...window, maximumParents: 3}, INITIAL, {
    maximumIterations: 1
  });
  expect(capped.candidateOverflow).toBe(3);
  expect(full.candidateOverflow).toBe(0);
  const parentTerm = (lag: number, distance: number) =>
    (0.5 / (2 * Math.PI)) * Math.exp(-lag - (distance * distance) / 2);
  // Last three events share parents at t=0 (inclusive lag=2) and t=1 (two rows).
  const backgroundFirstGroup = 1 + 0.1 / (0.1 + parentTerm(1, 1)) + 0.1 / (0.1 + parentTerm(1, 2));
  const cappedIntensity = 0.1 + parentTerm(1, 2) + parentTerm(1, 1);
  const fullIntensity = cappedIntensity + parentTerm(2, 3);
  expect(capped.estimates.background).toBeCloseTo(
    (backgroundFirstGroup + (3 * 0.1) / cappedIntensity) / 20,
    6
  );
  expect(full.estimates.background).toBeCloseTo(
    (backgroundFirstGroup + (3 * 0.1) / fullIntensity) / 20,
    6
  );
});

it('keeps large spatial variance in the log domain and scales its quotient by exponents', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const distance = Math.fround(1e19);
  const variance = Math.fround(distance * distance);
  const background = Math.fround(2e-38);
  const area = Math.fround(1e37);
  const result = await runHawkesGPU(
    device,
    [
      {x: 0, y: 0, time: 1},
      {x: distance, y: 0, time: 2}
    ],
    {...WINDOW, area, maximumDistance: 2 * distance},
    {...INITIAL, background, spatialStandardDeviation: distance},
    {maximumIterations: 1}
  );
  // 2*pi*variance overflows float32; 1/variance is subnormal. Neither intermediate
  // is needed to compute the finite normal estimates and dimensionless responsibilities.
  const parentIntensity = (0.5 / (2 * Math.PI * variance)) * Math.exp(-1.5);
  const triggered = parentIntensity / (background + parentIntensity);
  // One parent pair with lag 1 gives S=L=triggered, so the finite-window decay solves
  // 1/beta - 1 - sum(r exp(-beta r))/sum(1-exp(-beta r)) = 0 for remaining times r=[9,8]:
  // SciPy brentq gives beta=0.9980755456722574, M(beta)=1.9995337668598476.
  const mass = 1.9995337668598476;
  expect(result.estimates.background / ((2 - triggered) / (area * 10))).toBeCloseTo(1, 4);
  expect(result.estimates.branchingRatio / (triggered / mass)).toBeCloseTo(1, 4);
  expect(result.estimates.decayRate).toBeCloseTo(0.9980755456722574, 5);
  expect(result.estimates.spatialStandardDeviation / (distance * Math.SQRT1_2)).toBeCloseTo(1, 4);
});
