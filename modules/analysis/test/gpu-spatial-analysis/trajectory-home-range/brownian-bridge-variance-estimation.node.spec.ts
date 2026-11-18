// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';

it('refines the first positive interval even when zero beats its sampled endpoint', () => {
  const track = {x: [0, 500, 0, 1000, 0], y: [0, 500, 0, 1, 0], times: [0, 1, 2, 3, 4]};
  const options = {locationError: 500, windowSize: 5};
  const whole = estimateBrownianMotionVarianceWithStatus(track, options);
  const windows = estimateWindowedBrownianMotionVarianceWithStatus(track, options);
  for (const variance of [
    whole.motionVariance,
    ...windows.motionVariances,
    estimateBrownianMotionVariance(track, options),
    ...estimateWindowedBrownianMotionVariance(track, options)
  ]) {
    expect(Number.isFinite(variance)).toBe(true);
    expect(variance).toBeCloseTo(0.5, 5);
  }
  expect(whole.status.lowerBoundReached).toBe(false);
  expect(windows.segmentStatus.map(status => status.lowerBoundReached)).toEqual([
    false,
    false,
    false,
    false
  ]);
});

it.each([
  0, 0.01
])('keeps likelihood scoring work linear on smooth tracks, error %s', locationError => {
  const work: number[] = [];
  for (const count of [257, 513]) {
    const times = Array.from({length: count}, (_, row) => row);
    const track = {
      x: times.map(row => 10 * Math.sin(row * 0.1)),
      y: times.map(row => 10 * Math.cos(row * 0.1)),
      times
    };
    const logarithm = vi.spyOn(Math, 'log');
    try {
      const variance = estimateBrownianMotionVariance(track, {locationError});
      expect(Number.isFinite(variance)).toBe(true);
      const residualScale = 100 * (1 - Math.cos(0.1)) ** 2;
      expect(variance).toBeCloseTo(Math.max(0, residualScale - 3 * locationError ** 2), 10);
      work.push(logarithm.mock.calls.length);
    } finally {
      logarithm.mockRestore();
    }
  }
  if (locationError === 0) expect(work).toEqual([0, 0]);
  else {
    expect(work[0]).toBeGreaterThan(0);
    expect(work[1]).toBeLessThanOrEqual(work[0] * 2.5);
  }
});

it.each([false, true])('finds heterogeneous likelihood peaks, subnormal error: %s', subnormal => {
  const residual = subnormal ? 0 : 1e-7;
  const error = subnormal ? 1e-155 : 1e-8;
  const track = {x: [0, residual, 0, 2, 0], y: [0, 0, 0, 0, 0], times: [0, 1, 2, 3, 4]};
  const options = {locationError: [0, error, 0, 0.25, 0]};
  const likelihood = (motion: number) => {
    const smallVariance = 0.5 * motion + error ** 2;
    const largeVariance = 0.5 * motion + 0.0625;
    return (
      -Math.log(2 * Math.PI * smallVariance) -
      residual ** 2 / (2 * smallVariance) -
      Math.log(2 * Math.PI * largeVariance) -
      4 / (2 * largeVariance)
    );
  };
  const candidate = subnormal ? 0 : 9.8e-15;
  const whole = estimateBrownianMotionVarianceWithStatus(track, options);
  const windowed = estimateWindowedBrownianMotionVarianceWithStatus(track, options);
  for (const variance of [
    whole.motionVariance,
    ...windowed.motionVariances,
    estimateBrownianMotionVariance(track, options),
    ...estimateWindowedBrownianMotionVariance(track, options)
  ]) {
    expect(Number.isFinite(variance)).toBe(true);
    expect(likelihood(variance)).toBeGreaterThanOrEqual(likelihood(candidate) - 1e-12);
  }
  expect(whole.status.lowerBoundReached).toBe(subnormal);
  expect(windowed.segmentStatus.every(status => status.lowerBoundReached === subnormal)).toBe(true);
});

it('selects the higher boundary likelihood for heterogeneous location errors', () => {
  const track = {x: [0, 0, 0, 2, 0], y: [0, 0, 0, 0, 0], times: [0, 1, 2, 3, 4]};
  const result = estimateBrownianMotionVarianceWithStatus(track, {
    locationError: [0, 1e-9, 0, 0.25, 0]
  });
  const likelihood = (motion: number) =>
    [
      {variance: 0.5 * motion + 1e-18, residual: 0},
      {variance: 0.5 * motion + 0.0625, residual: 4}
    ].reduce(
      (total, term) =>
        total - Math.log(2 * Math.PI * term.variance) - term.residual / (2 * term.variance),
      0
    );
  expect(Number.isFinite(result.motionVariance)).toBe(true);
  expect(likelihood(result.motionVariance)).toBeGreaterThanOrEqual(likelihood(0));
  expect(result).toEqual({motionVariance: 0, status: {lowerBoundReached: true}});
  const windowed = estimateWindowedBrownianMotionVarianceWithStatus(track, {
    locationError: [0, 1e-9, 0, 0.25, 0]
  });
  expect(Array.from(windowed.motionVariances)).toEqual([0, 0, 0, 0]);
  expect(windowed.segmentStatus.every(status => status.lowerBoundReached)).toBe(true);
});
import {
  estimateBrownianMotionVariance,
  estimateBrownianMotionVarianceWithStatus,
  estimateWindowedBrownianMotionVariance,
  estimateWindowedBrownianMotionVarianceWithStatus
} from '../../../src/gpu-spatial-analysis/trajectory-home-range/brownian-bridge-variance-estimation';

it('includes the exact zero-motion likelihood maximum with nonzero residuals', () => {
  // Variance v = 1.5 + 0.5 m; likelihood derivative (0.5 - v) / v^2 < 0 for m >= 0.
  const track = {x: [0, 1, 0], y: [0, 0, 0], times: [0, 1, 2]};
  expect(estimateBrownianMotionVariance(track, {locationError: 1})).toBe(0);
  expect(Array.from(estimateWindowedBrownianMotionVariance(track, {locationError: 1}))).toEqual([
    0, 0
  ]);
  expect(estimateBrownianMotionVarianceWithStatus(track, {locationError: 1})).toEqual({
    motionVariance: 0,
    status: {lowerBoundReached: true}
  });
  expect(estimateWindowedBrownianMotionVarianceWithStatus(track, {locationError: 1})).toEqual({
    motionVariances: Float64Array.of(0, 0),
    segmentStatus: [{lowerBoundReached: true}, {lowerBoundReached: true}]
  });
});

it('retains the positive interior maximum', () => {
  // Squared residual 9 gives optimum v = 9 / 2, hence m = (4.5 - 1.5) / 0.5 = 6.
  const track = {x: [0, 3, 0], y: [0, 0, 0], times: [0, 1, 2]};
  expect(estimateBrownianMotionVariance(track, {locationError: 1})).toBeCloseTo(6, 5);
  for (const variance of estimateWindowedBrownianMotionVariance(track, {locationError: 1})) {
    expect(variance).toBeCloseTo(6, 5);
  }
  expect(estimateBrownianMotionVarianceWithStatus(track, {locationError: 1}).status).toEqual({
    lowerBoundReached: false
  });
});

it('does not label an unavailable estimate as a boundary optimum', () => {
  const result = estimateBrownianMotionVarianceWithStatus({x: [0], y: [0], times: [0]});
  expect(result.motionVariance).toBeNaN();
  expect(result.status.lowerBoundReached).toBe(false);
});

it('refines between zero and the first positive scan point', () => {
  // The exact optimum 4e-6 lies below the old scale * exp(-12) floor (about 9.2e-6).
  const track = {x: [0, Math.sqrt(3 + 4e-6), 0], y: [0, 0, 0], times: [0, 1, 2]};
  const estimate = estimateBrownianMotionVarianceWithStatus(track, {locationError: 1});
  // Likelihood differences here approach binary64 roundoff; this still excludes the old floor.
  expect(Number.isFinite(estimate.motionVariance)).toBe(true);
  expect(Math.abs(estimate.motionVariance - 4e-6)).toBeLessThan(2e-7);
  expect(estimate.status.lowerBoundReached).toBe(false);
});

it('distinguishes scaled zero optima and tiny positive optima without rounded likelihood ties', () => {
  for (const [middle, error, expected] of [
    [25, 25, 0],
    [Math.sqrt(3 + 1e-8), 1, Math.sqrt(3 + 1e-8) ** 2 - 3]
  ]) {
    const track = {x: [0, middle, 0], y: [0, 0, 0], times: [0, 1, 2]};
    const options = {locationError: error};
    const whole = estimateBrownianMotionVarianceWithStatus(track, options);
    const windowed = estimateWindowedBrownianMotionVarianceWithStatus(track, options);
    for (const value of [
      whole.motionVariance,
      estimateBrownianMotionVariance(track, options),
      ...windowed.motionVariances,
      ...estimateWindowedBrownianMotionVariance(track, options)
    ]) {
      if (expected === 0) expect(value).toBe(0);
      else expect(value).toBeCloseTo(expected, 12);
    }
    expect(whole.status.lowerBoundReached).toBe(expected === 0);
    expect(windowed.segmentStatus.map(status => status.lowerBoundReached)).toEqual([
      expected === 0,
      expected === 0
    ]);
  }
});

it.each([
  'coordinate',
  'locationError'
] as const)('reports an unavailable fit for a nonfinite %s', kind => {
  const track = {x: [0, kind === 'coordinate' ? NaN : 1, 0], y: [0, 0, 0], times: [0, 1, 2]};
  const options = {locationError: kind === 'locationError' ? [0, NaN, 0] : 1};
  expect(estimateBrownianMotionVarianceWithStatus(track, options)).toEqual({
    motionVariance: NaN,
    status: {lowerBoundReached: false}
  });
  const windowed = estimateWindowedBrownianMotionVarianceWithStatus(track, options);
  expect(Array.from(windowed.motionVariances)).toEqual([NaN, NaN]);
  expect(windowed.segmentStatus).toEqual([{lowerBoundReached: false}, {lowerBoundReached: false}]);
  expect(estimateBrownianMotionVariance(track, options)).toBeNaN();
  expect(Array.from(estimateWindowedBrownianMotionVariance(track, options))).toEqual([NaN, NaN]);
});

it('keeps a valid stationary fit at the lower bound', () => {
  expect(
    estimateBrownianMotionVarianceWithStatus(
      {x: [0, 0, 0], y: [0, 0, 0], times: [0, 1, 2]},
      {locationError: 1}
    )
  ).toEqual({motionVariance: 0, status: {lowerBoundReached: true}});
});

it('collects window diagnostics only when requested and preserves identical numeric columns', () => {
  const track = {x: [0, 1, 0, 3, 0], y: [0, 0, 0, 0, 0], times: [0, 1, 2, 3, 4]};
  const originalPush = Array.prototype.push;
  let collectedStatuses = 0;
  Array.prototype.push = function (this: unknown[], ...values: unknown[]): number {
    collectedStatuses += values.filter(
      value => typeof value === 'object' && value !== null && 'lowerBoundReached' in value
    ).length;
    return originalPush.apply(this, values);
  };
  let numeric: Float64Array;
  let diagnostic: ReturnType<typeof estimateWindowedBrownianMotionVarianceWithStatus>;
  let numericStatusCount: number;
  try {
    numeric = estimateWindowedBrownianMotionVariance(track, {locationError: 1});
    numericStatusCount = collectedStatuses;
    diagnostic = estimateWindowedBrownianMotionVarianceWithStatus(track, {locationError: 1});
  } finally {
    Array.prototype.push = originalPush;
  }
  expect(numericStatusCount).toBe(0);
  expect(collectedStatuses).toBe(track.times.length - 1);
  expect(numeric).toEqual(diagnostic.motionVariances);
});
