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
