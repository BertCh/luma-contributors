// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkTravelTimePercentilesOnCPU,
  CPUNetworkTravelTimePercentilesOperation
} from '../../../src/gpu-network/network-time-geography/cpu-network-travel-time-percentiles-operation';
import {
  GPUNetworkTravelTimePercentilesOperation,
  type NetworkTravelTimePercentilesOperationProps
} from '../../../src/gpu-network/network-time-geography/gpu-network-travel-time-percentiles-operation';
import {getNetworkTravelTimePercentileRank} from '../../../src/gpu-network/network-time-geography/network-travel-time-percentile-rank';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const INPUT = {
  stateVertices: new Uint32Array([0, 1, 1, 2]),
  transitions: new Uint32Array([0, 1, 0, 2]),
  costs: new Float32Array([2, 0.5, 20, 0]),
  initialState: new Uint32Array([0]),
  window: new Float32Array([0, 2]),
  vertexCount: 3,
  departureCount: 3,
  percentiles: [1, 50, 67, 100]
};
const PROPS: NetworkTravelTimePercentilesOperationProps = {
  id: 'percentile-test',
  stateCount: 4,
  transitionCount: 2,
  stateVertices: 'vertices',
  transitions: 'transitions',
  costs: 'costs',
  initialState: 'initial',
  window: 'window',
  vertexCount: 3,
  departureCount: 3,
  percentiles: [50, 100],
  output: {travelTimes: 'times', reachedCounts: 'reached', status: 'status'}
};

it('matches closed-form one-hop affine durations and hand-selected nearest ranks', () => {
  // t = 0,2,4 gives 2+t/2 = 2,3,4; the alternative state always takes 20.
  const result = computeNetworkTravelTimePercentilesOnCPU(INPUT);
  expect(result.travelTimes).toEqual(
    new Float32Array([0, 0, 0, 0, 2, 3, 4, 4, Infinity, Infinity, Infinity, Infinity])
  );
  expect(Array.from(result.reachedCounts)).toEqual([3, 3, 0]);
  expect(Array.from(result.status)).toEqual([1, 1, 1, 1, 1, 1]);
});

it('distinguishes percentiles below, at and above the median without an epsilon', () => {
  const result = computeNetworkTravelTimePercentilesOnCPU({
    ...INPUT,
    stateVertices: Uint32Array.of(0, 1),
    transitions: Uint32Array.of(0, 1),
    // Departures 0 and 1 have travel times 5 and 90.
    costs: Float32Array.of(5, 85),
    window: Float32Array.of(0, 1),
    departureCount: 2,
    percentiles: [49.99999999, 50, 50.00000001, 100]
  });
  expect(Array.from(result.travelTimes.subarray(4, 8))).toEqual([5, 5, 90, 90]);
  expect(Array.from(result.status)).toEqual([1, 1, 1, 1]);
});

it('keeps integer percentage boundaries exact despite quotient rounding', () => {
  const result = computeNetworkTravelTimePercentilesOnCPU({
    ...INPUT,
    stateVertices: Uint32Array.of(0, 1),
    transitions: Uint32Array.of(0, 1),
    costs: Float32Array.of(1, 1),
    window: Float32Array.of(0, 1),
    departureCount: 25,
    // Sorted durations are 1..25; the adjacent binary64 value above 56% must select 15.
    percentiles: [28, 56 - 2 ** -47, 56, 56 + 2 ** -47, 100]
  });
  expect(Array.from(result.travelTimes.subarray(5, 10))).toEqual([7, 14, 14, 15, 25]);
});

it.each([
  {departureCount: 3, percentiles: [100 / 3], expected: [2]},
  {
    departureCount: 2,
    percentiles: [Number.MIN_VALUE, 50 - 2 ** -47, 50, 50 + 2 ** -47, 100],
    expected: [1, 1, 1, 2, 2]
  }
])('uses the exact supplied binary64 percentile for $departureCount departures', fixture => {
  const result = computeNetworkTravelTimePercentilesOnCPU({
    ...INPUT,
    ...fixture,
    stateVertices: Uint32Array.of(0, 1),
    transitions: Uint32Array.of(0, 1),
    costs: Float32Array.of(1, 1),
    window: Float32Array.of(0, 1)
  });
  const columnCount = fixture.percentiles.length;
  expect(Array.from(result.travelTimes.subarray(columnCount, 2 * columnCount))).toEqual(
    fixture.expected
  );
});

it('reads a typed decimal percentile as written (NET-102)', () => {
  // Departure k of 0..n-1 has travel time k + 1, so rank r selects r. Nearest rank in decimal
  // arithmetic: ceil(p * n / 100) gives 0.1% -> 1, 12.3% -> 123, 99.9% -> 999 of 1,000 and
  // 99.9% -> 1,998 of 2,000. The binary64 values of 0.1, 12.3 and 99.9 lie just above the
  // decimals, which previously selected 2, 124, 1,000 and 1,999.
  for (const {departureCount, percentiles, expected} of [
    {departureCount: 1000, percentiles: [0.1, 12.3, 99.9, 100], expected: [1, 123, 999, 1000]},
    {departureCount: 2000, percentiles: [99.9], expected: [1998]}
  ]) {
    const result = computeNetworkTravelTimePercentilesOnCPU({
      ...INPUT,
      stateVertices: Uint32Array.of(0, 1),
      transitions: Uint32Array.of(0, 1),
      costs: Float32Array.of(1, 1),
      window: Float32Array.of(0, 1),
      departureCount,
      percentiles
    });
    const columnCount = percentiles.length;
    // Vertex 0 is the origin, vertex 1 has the selected ranks, and vertex 2 is unreachable.
    expect(Array.from(result.travelTimes)).toEqual([
      ...new Array<number>(columnCount).fill(0),
      ...expected,
      ...new Array<number>(columnCount).fill(Infinity)
    ]);
  }
});

it('matches the decimal nearest rank for every percentile with up to three decimals (NET-102)', () => {
  // Independent closed form: percentile k / 1000 of n departures has rank
  // ceil(k * n / 100000), in integers.
  const mismatches: string[] = [];
  for (const departureCount of [7, 60, 1000, 2000]) {
    for (let thousandths = 1; thousandths <= 100000; thousandths += 1) {
      const percentile = thousandths / 1000;
      const expected = Number((BigInt(thousandths) * BigInt(departureCount) + 99999n) / 100000n);
      const rank = getNetworkTravelTimePercentileRank(percentile, departureCount);
      if (rank !== expected) {
        mismatches.push(`${percentile} of ${departureCount}: ${rank} instead of ${expected}`);
      }
    }
  }
  expect(mismatches.slice(0, 5)).toEqual([]);
  // Adjacent binary64 values keep their own ranks (the earlier 1e-9 slack defect).
  expect(getNetworkTravelTimePercentileRank(56 + 2 ** -47, 25)).toBe(15);
  expect(getNetworkTravelTimePercentileRank(56 - 2 ** -47, 25)).toBe(14);
  expect(getNetworkTravelTimePercentileRank(50.00000001, 2)).toBe(2);
  expect(getNetworkTravelTimePercentileRank(Number.MIN_VALUE, 2)).toBe(1);
  expect(getNetworkTravelTimePercentileRank(1e-7, 1000)).toBe(1);
});

it('sorts waiting travel times, counts invalid runs and keeps subtraction in float64', () => {
  const piecewise = computeNetworkTravelTimePercentilesOnCPU({
    ...INPUT,
    stateVertices: new Uint32Array([0, 1]),
    transitions: new Uint32Array([0, 1]),
    costs: new Float32Array([5, 9, 10, 12]),
    costModel: 'piecewise-arrival',
    samplesPerTransition: 2,
    window: new Float32Array([0, 5]),
    percentiles: [1, 50, 100]
  });
  expect(Array.from(piecewise.travelTimes.subarray(3, 6))).toEqual([2, 4, 9]);
  const invalid = computeNetworkTravelTimePercentilesOnCPU({
    ...INPUT,
    window: new Float32Array([1, -1]),
    percentiles: [50, 100]
  });
  expect(Array.from(invalid.status)).toEqual([1, 1, 1, 1, 0, 0]);
  expect(Array.from(invalid.reachedCounts)).toEqual([2, 2, 0]);
  expect(Array.from(invalid.travelTimes)).toEqual([0, Infinity, 2.5, Infinity, Infinity, Infinity]);
  const precise = computeNetworkTravelTimePercentilesOnCPU({
    ...INPUT,
    stateVertices: new Uint32Array([0, 1]),
    transitions: new Uint32Array([0, 1]),
    costs: new Float32Array([1, 0]),
    window: new Float32Array([2 ** 24, 0]),
    departureCount: 1,
    percentiles: [100]
  });
  expect(precise.travelTimes[1]).toBe(1);
});

it('rejects inconsistent arrays and validates both operation constructions', () => {
  expect(() =>
    computeNetworkTravelTimePercentilesOnCPU({...INPUT, window: new Float32Array(1)})
  ).toThrow(/window/);
  expect(() =>
    computeNetworkTravelTimePercentilesOnCPU({...INPUT, initialState: new Uint32Array(2)})
  ).toThrow(/initialState/);
  expect(() =>
    computeNetworkTravelTimePercentilesOnCPU({...INPUT, costs: new Float32Array(1)})
  ).toThrow(/costs/);
  const invalid = {...PROPS, percentiles: [90, 50]};
  expect(() => new CPUNetworkTravelTimePercentilesOperation(invalid)).toThrow(
    /percentile-test.*percentiles/
  );
  expect(() => new CPUNetworkTravelTimePercentilesOperation({...PROPS, departureCount: 0})).toThrow(
    /percentile-test.*departureCount/
  );
  const shapes = {
    stateVertices: {format: 'uint32' as const, length: 4},
    transitions: {format: 'uint32x2' as const, length: 2},
    costs: {format: 'float32x2' as const, length: 2},
    initialState: {format: 'uint32' as const, length: 1},
    window: {format: 'float32' as const, length: 2}
  };
  expect(() =>
    constructNetworkOperationContributor(
      new GPUNetworkTravelTimePercentilesOperation(invalid),
      shapes
    )
  ).toThrow(/percentile-test.*percentiles/);
  expect(() =>
    constructNetworkOperationContributor(
      new GPUNetworkTravelTimePercentilesOperation(PROPS),
      shapes
    )
  ).not.toThrow();
  for (const Operation of [
    GPUNetworkTravelTimePercentilesOperation,
    CPUNetworkTravelTimePercentilesOperation
  ])
    expect(() => new Operation({...PROPS, output: {travelTimes: 'window'}})).toThrow(
      /percentile-test/
    );
  expect(
    () =>
      new CPUNetworkTravelTimePercentilesOperation({
        ...PROPS,
        departureCount: 61,
        percentiles: new Array<number>(17).fill(50),
        stateCount: 257,
        transitionCount: 4097
      })
  ).not.toThrow();
});

it('preserves pure CPU sample counts above uint32 when no transitions need samples (S1)', () => {
  // The only state is the origin; no edge reads a sample, so every duration is exactly zero.
  const result = computeNetworkTravelTimePercentilesOnCPU({
    stateVertices: Uint32Array.of(0),
    transitions: new Uint32Array(0),
    costs: new Float32Array(0),
    costModel: 'piecewise-arrival',
    samplesPerTransition: 2 ** 32,
    vertexCount: 1,
    initialState: Uint32Array.of(0),
    window: Float32Array.of(0, 1),
    departureCount: 1,
    percentiles: [100]
  });
  expect(result.travelTimes).toEqual(Float32Array.of(0));
  expect(result.reachedCounts).toEqual(Uint32Array.of(1));
  expect(result.status).toEqual(Uint32Array.of(1, 1));
});

it('NAM-01 treats .95 and 95 as percentages', () => {
  const result = computeNetworkTravelTimePercentilesOnCPU({
    ...INPUT,
    stateVertices: Uint32Array.of(0, 1),
    transitions: Uint32Array.of(0, 1),
    costs: Float32Array.of(5, 85),
    window: Float32Array.of(0, 1),
    departureCount: 2,
    percentiles: [0.95, 95]
  });
  expect(Array.from(result.travelTimes.slice(2, 4))).toEqual([5, 90]);
});

it('STA-14 distinguishes reached overflow from unreachable and ordinary finite costs', () => {
  for (const cost of [3e38, 7]) {
    const result = computeNetworkTravelTimePercentilesOnCPU({
      ...INPUT,
      stateVertices: Uint32Array.of(0, 1, 2),
      transitions: Uint32Array.of(0, 1, 1, 2),
      costs: Float32Array.of(cost, 0, cost, 0),
      window: Float32Array.of(0, 1),
      departureCount: 1,
      percentiles: [50]
    });
    expect(Array.from(result.reachedCounts)).toEqual([1, 1, 1]);
    expect(Array.from(result.status)).toEqual([1, 1]);
    expect(Array.from(result.unrepresentable)).toEqual([0, 0, cost === 7 ? 0 : 1]);
    expect(result.travelTimes[2]).toBe(cost === 7 ? 14 : Infinity);
  }
});

it('UNI-16 preserves affine duration only when departure and intercept are rebased together', () => {
  const input = {
    ...INPUT,
    stateVertices: Uint32Array.of(0, 1),
    transitions: Uint32Array.of(0, 1),
    departureCount: 1,
    percentiles: [50]
  };
  for (const [departure, base] of [
    [100, 10],
    [0, 60]
  ]) {
    const result = computeNetworkTravelTimePercentilesOnCPU({
      ...input,
      window: Float32Array.of(departure, 0),
      costs: Float32Array.of(base, 0.5)
    });
    // d(100) = 10 + .5 * 100 = 60; d′(0) = (10 + .5 * 100) + .5 * 0.
    expect(result.travelTimes[1]).toBe(60);
  }
});

it('FLT-12 retains short elapsed routes at departure 16777216 on the CPU', () => {
  for (const direct of [false, true]) {
    const result = computeNetworkTravelTimePercentilesOnCPU({
      ...INPUT,
      stateVertices: Uint32Array.of(0, 1, 2),
      transitions: direct ? Uint32Array.of(0, 1, 1, 2, 0, 2) : Uint32Array.of(0, 1, 1, 2),
      costs: direct ? Float32Array.of(1, 0, 1, 0, 1.5, 0) : Float32Array.of(1, 0, 1, 0),
      departureCount: 1,
      percentiles: [50],
      window: Float32Array.of(16777216, 0)
    });
    // Literal shortest paths: 0; 1; min(1+1, optional direct 1.5).
    expect(Array.from(result.travelTimes)).toEqual([0, 1, direct ? 1.5 : 2]);
    expect(Array.from(result.reachedCounts)).toEqual([1, 1, 1]);
    expect(Array.from(result.status)).toEqual([1, 1]);
  }
});

it('review B1 preserves elapsed CPU paths beyond the float64 absolute-clock spacing', () => {
  const result = computeNetworkTravelTimePercentilesOnCPU({
    ...INPUT,
    stateVertices: Uint32Array.of(0, 1, 2),
    transitions: Uint32Array.of(0, 1, 1, 2, 0, 2),
    costs: Float32Array.of(1, 0, 1, 0, 1.5, 0),
    departureCount: 1,
    percentiles: [50],
    window: Float32Array.of(2 ** 54, 0)
  });
  expect(Array.from(result.travelTimes)).toEqual([0, 1, 1.5]);
});
