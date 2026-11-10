// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {runHawkesGPU} from './stm08-hawkes-harness';
import {
  fitHawkesReference,
  measureHawkesSamplingDistribution,
  simulateHawkes,
  type HawkesParameters,
  type HawkesWindow
} from './stm08-hawkes-reference';

// Oracle: the binary64 EM of stm08-hawkes-reference.ts. The ETAS R package (0.7.2, GPL) is the
// oracle the review names; R is not installed here, so that fixture is PENDING and nothing below
// claims agreement with ETAS.

const TRUTH: HawkesParameters = {
  background: 2,
  branchingRatio: 0.5,
  decayRate: 2,
  spatialStandardDeviation: 0.02
};
const WINDOW: HawkesWindow = {
  area: 1,
  duration: 400,
  maximumLag: 5,
  maximumDistance: 0.12,
  maximumParents: 64
};
const INITIAL: HawkesParameters = {
  background: 1,
  branchingRatio: 0.3,
  decayRate: 1,
  spatialStandardDeviation: 0.05
};
const NAMES = ['background', 'branchingRatio', 'decayRate', 'spatialStandardDeviation'] as const;

/**
 * Measured float32 against binary64 on the same catalogue after the same number of iterations:
 * 7e-7 relative in the estimates after 1 to 3 iterations (STM-08 working notes), and the
 * log-likelihood to 4e-7. 1e-5 leaves a margin of more than ten for another device.
 */
const SAME_ITERATION_RELATIVE_TOLERANCE = 1e-5;
/**
 * At convergence the stopping iteration may differ by one between precisions, and the last step
 * moves the estimates by up to the 1e-4 stopping tolerance times the contraction rate; measured
 * differences were 3e-5 at most.
 */
const CONVERGED_RELATIVE_TOLERANCE = 3e-4;

function expectEstimates(
  actual: HawkesParameters,
  expected: HawkesParameters,
  relativeTolerance: number,
  label: string
) {
  for (const name of NAMES) {
    expect(
      Math.abs(actual[name] - expected[name]) / expected[name],
      `${label} ${name}: ${actual[name]} vs ${expected[name]}`
    ).toBeLessThan(relativeTolerance);
  }
}

it('GPUSpaceTimeHawkes equals the binary64 EM after the same number of iterations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {events} = simulateHawkes(TRUTH, 1, WINDOW.duration, 7);
  for (const iterations of [1, 3, 5]) {
    const reference = fitHawkesReference(events, WINDOW, INITIAL, {
      maximumIterations: iterations,
      tolerance: 1e-12
    });
    const gpu = await runHawkesGPU(device, events, WINDOW, INITIAL, {
      maximumIterations: iterations,
      tolerance: 1e-12
    });
    expectEstimates(
      gpu.estimates,
      reference.estimates,
      SAME_ITERATION_RELATIVE_TOLERANCE,
      `${iterations} iterations`
    );
    expect(Math.abs(gpu.logLikelihood - reference.logLikelihood)).toBeLessThan(
      SAME_ITERATION_RELATIVE_TOLERANCE * Math.abs(reference.logLikelihood)
    );
    // Budget exhausted: the latest iterate is published and the flags say so.
    expect(gpu.iterationCount).toBe(iterations);
    expect(gpu.converged).toBe(0);
    expect(gpu.iterationLimitReached).toBe(1);
  }
}, 120000);

it('GPUSpaceTimeHawkes converges to the binary64 fit and reports convergence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const seed of [7, 8]) {
    const {events} = simulateHawkes(TRUTH, 1, WINDOW.duration, seed);
    const reference = fitHawkesReference(events, WINDOW, INITIAL, {
      maximumIterations: 100,
      tolerance: 1e-4
    });
    const gpu = await runHawkesGPU(device, events, WINDOW, INITIAL, {maximumIterations: 100});
    expect(reference.converged).toBe(true);
    expect(gpu.converged).toBe(1);
    expect(gpu.iterationLimitReached).toBe(0);
    expect(Math.abs(gpu.iterationCount - reference.iterationCount)).toBeLessThanOrEqual(1);
    expect(gpu.iterationCount).toBeLessThan(100);
    expectEstimates(
      gpu.estimates,
      reference.estimates,
      CONVERGED_RELATIVE_TOLERANCE,
      `seed ${seed}`
    );
    expect(gpu.candidateOverflow).toBe(0);
    expect(gpu.invalidCount).toBe(0);
    expect(gpu.approximation).toBe(1);
    expect(Math.abs(gpu.logLikelihood - reference.logLikelihood)).toBeLessThan(
      1e-4 * Math.abs(reference.logLikelihood)
    );
  }
}, 120000);

it('GPUSpaceTimeHawkes recovers the generating parameters within their sampling variability', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Sampling distribution of the EM estimator at this configuration, from 24 independent
  // simulations fitted in binary64. A recovered estimate must lie within the estimator's bias plus
  // four standard deviations of the truth, a two-sided normal probability of about 6e-5 per
  // parameter. The bias (mean minus truth) is the unmodelled spatial edge, about 3 percent of the
  // offspring near the border, and the finite parent bounds.
  const distribution = measureHawkesSamplingDistribution(TRUTH, 1, WINDOW, INITIAL, 24, 1000, {
    maximumIterations: 100,
    tolerance: 1e-4
  });
  for (const name of NAMES) {
    // Teeth: the spread is a real fraction of the parameter, so the check is not vacuous, and
    // narrow enough that a wrong model (see the negative controls) falls outside.
    expect(distribution.standardDeviation[name]).toBeGreaterThan(0);
    expect(distribution.standardDeviation[name] / TRUTH[name]).toBeLessThan(0.1);
  }
  for (const seed of [7, 8, 9]) {
    const {events} = simulateHawkes(TRUTH, 1, WINDOW.duration, seed);
    const gpu = await runHawkesGPU(device, events, WINDOW, INITIAL, {maximumIterations: 100});
    expect(gpu.converged).toBe(1);
    for (const name of NAMES) {
      const bias = Math.abs(distribution.mean[name] - TRUTH[name]);
      const tolerance = bias + 4 * distribution.standardDeviation[name];
      expect(
        Math.abs(gpu.estimates[name] - TRUTH[name]),
        `seed ${seed} ${name}: ${gpu.estimates[name]} vs ${TRUTH[name]} (tolerance ${tolerance})`
      ).toBeLessThan(tolerance);
    }
  }
}, 240000);

it('GPUSpaceTimeHawkes separates triggered from background events and a Poisson catalogue from a clustered one', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const clustered = simulateHawkes(TRUTH, 1, WINDOW.duration, 11);
  const fit = await runHawkesGPU(device, clustered.events, WINDOW, INITIAL, {
    maximumIterations: 100
  });
  let backgroundMean = 0;
  let backgroundCount = 0;
  let offspringMean = 0;
  let offspringCount = 0;
  clustered.isBackground.forEach((isBackground, row) => {
    if (isBackground) {
      backgroundMean += fit.backgroundProbability[row];
      backgroundCount++;
    } else {
      offspringMean += fit.backgroundProbability[row];
      offspringCount++;
    }
  });
  backgroundMean /= backgroundCount;
  offspringMean /= offspringCount;
  // Generating labels are unseen by the fit; the model's own overlap (offspring born next to
  // nothing in time, background events landing near another) bounds the separation below 1.
  expect(backgroundMean).toBeGreaterThan(0.85);
  expect(offspringMean).toBeLessThan(0.25);

  // Negative control: no clustering. The same fit must return a branching ratio near zero and
  // call (almost) every event background, whereas the clustered catalogue gives about one half.
  const poisson = simulateHawkes({...TRUTH, branchingRatio: 0}, 1, WINDOW.duration, 12);
  const poissonFit = await runHawkesGPU(device, poisson.events, WINDOW, INITIAL, {
    maximumIterations: 100
  });
  expect(poissonFit.estimates.branchingRatio).toBeLessThan(0.1);
  expect(fit.estimates.branchingRatio).toBeGreaterThan(0.4);
  expect(poissonFit.estimates.background).toBeGreaterThan(0.85 * TRUTH.background);
}, 120000);

it('GPUSpaceTimeHawkes counts events whose parent window was cut', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {events} = simulateHawkes(TRUTH, 1, WINDOW.duration, 7);
  for (const maximumParents of [1, 4, 64]) {
    const window = {...WINDOW, maximumParents};
    const reference = fitHawkesReference(events, window, INITIAL, {
      maximumIterations: 2,
      tolerance: 1e-12
    });
    const gpu = await runHawkesGPU(device, events, window, INITIAL, {
      maximumIterations: 2,
      tolerance: 1e-12
    });
    expect(gpu.candidateOverflow, `maximumParents ${maximumParents}`).toBe(reference.cutCount);
    expectEstimates(gpu.estimates, reference.estimates, 1e-4, `maximumParents ${maximumParents}`);
  }
  // Teeth: the counts are not all zero, and a larger capacity cuts fewer.
  const small = await runHawkesGPU(device, events, {...WINDOW, maximumParents: 1}, INITIAL, {
    maximumIterations: 1
  });
  expect(small.candidateOverflow).toBeGreaterThan(100);
}, 120000);

it('GPUSpaceTimeHawkes does not let simultaneous events trigger each other', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Every event has a twin at the same time and place: a lag of zero must not count as a parent.
  const {events} = simulateHawkes(TRUTH, 1, WINDOW.duration, 13);
  const twinned = events.slice(0, 400).flatMap(event => [event, {...event}]);
  const reference = fitHawkesReference(twinned, WINDOW, INITIAL, {
    maximumIterations: 3,
    tolerance: 1e-12
  });
  const gpu = await runHawkesGPU(device, twinned, WINDOW, INITIAL, {
    maximumIterations: 3,
    tolerance: 1e-12
  });
  expectEstimates(gpu.estimates, reference.estimates, SAME_ITERATION_RELATIVE_TOLERANCE, 'twins');
}, 60000);

it('GPUSpaceTimeHawkes excludes rows outside the window and with non-finite values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {events} = simulateHawkes(TRUTH, 1, WINDOW.duration, 7);
  const clean = await runHawkesGPU(device, events, WINDOW, INITIAL, {maximumIterations: 100});
  const dirty = [
    {x: 0.5, y: 0.5, time: Number.NaN},
    ...events.slice(0, 40),
    {x: 0.5, y: 0.5, time: WINDOW.duration + 1},
    ...events.slice(40),
    {x: Number.NaN, y: 0.5, time: 3},
    {x: 0.5, y: 0.5, time: -2}
  ];
  const withInvalid = await runHawkesGPU(device, dirty, WINDOW, INITIAL, {maximumIterations: 100});
  expect(withInvalid.invalidCount).toBe(4);
  expectEstimates(withInvalid.estimates, clean.estimates, 1e-6, 'invalid rows ignored');
  [0, 41, dirty.length - 2, dirty.length - 1].forEach(row => {
    expect(withInvalid.backgroundProbability[row]).toBeNaN();
  });
  expect(Number.isFinite(withInvalid.backgroundProbability[1])).toBe(true);
  // Teeth for the channel itself: a catalogue with every row excluded has nothing to fit.
  const empty = await runHawkesGPU(
    device,
    events.map(event => ({...event, time: -1})),
    WINDOW,
    INITIAL,
    {
      maximumIterations: 5
    }
  );
  expect(empty.invalidCount).toBe(events.length);
  expect(empty.estimates.background).toBeNaN();
  expect(empty.converged).toBe(0);
  expect(empty.iterationLimitReached).toBe(0);
}, 120000);

it('GPUSpaceTimeHawkes reports an unusable start without fitting', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {events} = simulateHawkes(TRUTH, 1, WINDOW.duration, 7);
  const result = await runHawkesGPU(device, events, WINDOW, INITIAL, {
    maximumIterations: 5,
    parameterOverride: {5: -1}
  });
  expect(result.estimates.background).toBeNaN();
  expect(result.estimates.decayRate).toBeNaN();
  expect(result.iterationCount).toBe(0);
  expect(result.converged).toBe(0);
  expect(result.iterationLimitReached).toBe(0);
});

it('GPUSpaceTimeHawkes is bitwise repeatable', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {events} = simulateHawkes(TRUTH, 1, WINDOW.duration, 9);
  const first = await runHawkesGPU(device, events, WINDOW, INITIAL, {maximumIterations: 100});
  const second = await runHawkesGPU(device, events, WINDOW, INITIAL, {maximumIterations: 100});
  expect(second.estimates).toEqual(first.estimates);
  expect(second.logLikelihood).toBe(first.logLikelihood);
  expect(second.backgroundProbability).toEqual(first.backgroundProbability);
}, 120000);
