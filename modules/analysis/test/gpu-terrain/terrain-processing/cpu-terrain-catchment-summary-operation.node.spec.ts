// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTerrainCatchmentSummaryOnCPU,
  CPUTerrainCatchmentSummaryOperation
} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation';
import {GPUTerrainCatchmentSummaryOperation} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  makeCatchmentInput,
  makeCatchmentProps
} from './terrain-catchment-summary-operation-fixture';

it('matches a hand-counted two-basin histogram, cross-basin edges and stream profiles', () => {
  const result = computeTerrainCatchmentSummaryOnCPU(makeCatchmentInput(), makeCatchmentProps());
  // Basin 0: heights {8,10,16,20}, four 6-unit cells; basin 1: {4,6,8,12}.
  expect(Array.from(result.statistics)).toEqual([
    24,
    13.5,
    8,
    20,
    6,
    0.25,
    24,
    7.5,
    4,
    12,
    2,
    Math.fround(1 / 12),
    0,
    NaN,
    NaN,
    NaN,
    0,
    NaN
  ]);
  expect(Array.from(result.hypsometry)).toEqual([12, 12, 18, 24, 12, 18, 18, 24, 0, 0, 0, 0]);
  expect(Array.from(result.profiles)).toEqual([
    6,
    10,
    1,
    4,
    8,
    1,
    2,
    6,
    1,
    0,
    4,
    0,
    NaN,
    20,
    NaN,
    2,
    16,
    2,
    0,
    12,
    0,
    NaN,
    8,
    NaN
  ]);
  expect(Array.from(result.profileStatus)).toEqual([1, 1, 1, 1, 0, 1, 1, 0]);
});

it('retains the exact residual of opposite large elevations in both cancellation orders', () => {
  const magnitude = 2 ** 60;
  const props = makeCatchmentProps();
  const options = {
    ...props,
    grid: {...props.grid, width: 3, height: 1, affine: [1, 0, 0, 0, -1, 0] as const},
    basinCount: 1,
    histogramBins: 2
  };
  for (const elevations of [
    [magnitude, 1, -magnitude],
    [magnitude, -magnitude, 1]
  ]) {
    const input = {
      elevation: Float32Array.from(elevations),
      basinIds: new Uint32Array(3),
      streams: new Uint32Array(3),
      flowDirections: new Uint32Array(3)
    };
    const result = computeTerrainCatchmentSummaryOnCPU(input, options);
    // Every input is exactly representable in float32. The exact integer sum is 1,
    // so the three-cell mean is 1/3, independently of where the cancellation occurs.
    expect(Array.from(result.statistics)).toEqual([
      3,
      Math.fround(1 / 3),
      -magnitude,
      magnitude,
      0,
      0
    ]);
    expect(Array.from(result.hypsometry)).toEqual([1, 3]);
    expect(Array.from(result.profileStatus)).toEqual([0, 0, 0]);
    expect(Array.from(result.profiles)).toEqual(elevations.flatMap(value => [NaN, value, NaN]));
  }
});

it('copies close positive basin extrema without an arithmetic tolerance', () => {
  const props = makeCatchmentProps();
  const maximum = Math.fround(8.0001);
  const input = {
    elevation: Float32Array.of(8, maximum, 8),
    basinIds: new Uint32Array(3),
    streams: new Uint32Array(3),
    flowDirections: new Uint32Array(3)
  };
  const result = computeTerrainCatchmentSummaryOnCPU(input, {
    ...props,
    grid: {...props.grid, width: 3, height: 1, affine: [1, 0, 0, 0, -1, 0] as const},
    basinCount: 1,
    histogramBins: 2
  });
  // The minimum is the copied 8 and the maximum is the distinct, already rounded input.
  expect(result.statistics[2]).toBe(8);
  expect(result.statistics[3]).toBe(maximum);
  expect(result.statistics[2]).not.toBe(maximum);
});

it('keeps exclusive step bounds, cycles, unresolved words and invalid/excluded starts', () => {
  const input = makeCatchmentInput(),
    props = makeCatchmentProps();
  expect(
    Array.from(
      computeTerrainCatchmentSummaryOnCPU(input, {...props, maximumSteps: 2}).profileStatus
    )
  ).toEqual([2, 2, 1, 1, 0, 1, 1, 0]);
  input.flowDirections[1] = 16;
  expect(
    Array.from(computeTerrainCatchmentSummaryOnCPU(input, props).profileStatus).slice(0, 4)
  ).toEqual([2, 2, 1, 1]);
  input.flowDirections[1] = 1;
  input.basinIds[1] = 0xffffffff;
  let result = computeTerrainCatchmentSummaryOnCPU(input, props);
  expect(Array.from(result.profileStatus).slice(0, 4)).toEqual([1, 0, 1, 1]);
  expect(result.profiles[0]).toBe(6); // Excluded intermediate basin IDs do not stop a profile.
  input.streams[2] = 0xfffffffe;
  input.validity[3] = 0;
  result = computeTerrainCatchmentSummaryOnCPU(input, props);
  expect(Array.from(result.profileStatus).slice(0, 4)).toEqual([2, 0, 2, 0]);
  expect(result.profiles[10]).toBeNaN();
  input.flowDirections[0] = 64; // Nonzero code with no in-grid receiver is unresolved, not terminal.
  expect(computeTerrainCatchmentSummaryOnCPU(input, props).profileStatus[0]).toBe(2);
});

it('handles constant and negative basin heights, every encoding and geographic cell area', () => {
  const input = makeCatchmentInput(),
    props = makeCatchmentProps();
  input.elevation.fill(-2);
  const result = computeTerrainCatchmentSummaryOnCPU(input, props);
  expect(Array.from(result.hypsometry)).toEqual([24, 24, 24, 24, 24, 24, 24, 24, 0, 0, 0, 0]);
  for (const directionEncoding of ['whitebox', 'taudem'] as const) {
    const encoded = {
      ...input,
      flowDirections: Uint32Array.from(input.flowDirections, word =>
        word === 0 ? 0 : directionEncoding === 'whitebox' ? 2 : 1
      )
    };
    expect(computeTerrainCatchmentSummaryOnCPU(encoded, {...props, directionEncoding})).toEqual(
      result
    );
  }
  const geographic = {
    ...props,
    grid: {...props.grid, affine: [1, 0, 0, 0, -1, 1] as const},
    cellSizeMode: 'geographic' as const
  };
  const areas = computeTerrainCatchmentSummaryOnCPU(input, geographic).statistics;
  expect(areas[0]).toBeGreaterThan(4.9e10);
  expect(areas[0]).toBeLessThan(4.94e10);
  expect(areas[0]).toBe(areas[6]); // The basins each span two cells at the same two latitudes.
});

it('constructs the deferred GPU contributor and checks its static validation', () => {
  const props = makeCatchmentProps();
  const shapes = {
    elevation: {format: 'float32' as const, length: 8},
    validity: {format: 'uint32' as const, length: 8},
    basinIds: {format: 'uint32' as const, length: 8},
    streams: {format: 'uint32' as const, length: 8},
    flowDirections: {format: 'uint32' as const, length: 8}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUTerrainCatchmentSummaryOperation(props), shapes)
  ).not.toThrow();
  expect(() =>
    constructNetworkOperationContributor(
      new GPUTerrainCatchmentSummaryOperation({...props, maximumSteps: 0}),
      shapes
    )
  ).toThrow(/catchment-test.*maximumSteps/);
});

it('validates both operation shapes, CPU options and every array length', () => {
  const props = makeCatchmentProps();
  for (const Operation of [
    GPUTerrainCatchmentSummaryOperation,
    CPUTerrainCatchmentSummaryOperation
  ]) {
    expect(() => new Operation({...props, histogramBins: 0})).toThrow(/^catchment-test/);
    expect(() => new Operation({...props, basinCount: 0})).toThrow(/^catchment-test/);
    expect(
      () => new Operation({...props, output: {...props.output, profiles: 'elevation'}})
    ).toThrow(/^catchment-test/);
  }
  expect(() => new CPUTerrainCatchmentSummaryOperation({...props, maximumSteps: 0})).toThrow(
    /catchment-test.*maximumSteps/
  );
  expect(
    () =>
      new CPUTerrainCatchmentSummaryOperation({
        ...props,
        grid: {...props.grid, affine: [1, 0, 0, 0, 0, 0]}
      })
  ).toThrow(/catchment-test.*determinant/);
  const input = makeCatchmentInput();
  for (const name of ['elevation', 'basinIds', 'streams', 'flowDirections', 'validity'] as const) {
    expect(() =>
      computeTerrainCatchmentSummaryOnCPU({...input, [name]: input[name].subarray(1)}, props)
    ).toThrow(new RegExp(name));
  }
});
