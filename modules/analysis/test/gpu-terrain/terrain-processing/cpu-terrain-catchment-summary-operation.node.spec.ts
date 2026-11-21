// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUTerrainCatchmentSummaryOperation as PublicCPUTerrainCatchmentSummaryOperation,
  type CPUTerrainCatchmentSummaryOperationProps
} from '@luma-contributors/analysis/gpu-terrain';
import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeTerrainCatchmentSummaryOnCPU,
  CPUTerrainCatchmentSummaryOperation
} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation';
import {GPUTerrainCatchmentSummaryOperation} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  makeCatchmentStatusCases,
  expectCatchmentStatusReadbacks,
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

it('declares and binds omitted, solitary and combined aggregate status names', () => {
  const base = makeCatchmentProps();
  const shapes = {
    elevation: {format: 'float32' as const, length: 8},
    validity: {format: 'uint32' as const, length: 8},
    basinIds: {format: 'uint32' as const, length: 8},
    streams: {format: 'uint32' as const, length: 8},
    flowDirections: {format: 'uint32' as const, length: 8}
  };
  const all = {converged: 'done', invalidCount: 'invalid', iterationLimitReached: 'limited'};
  for (const status of [
    undefined,
    ...Object.entries(all).map(([key, value]) => ({[key]: value})),
    all
  ]) {
    const props = {...base, output: {...base.output, status}};
    const operation = new GPUTerrainCatchmentSummaryOperation(props);
    expect(Object.keys(operation.outputs)).toHaveLength(4 + Object.keys(status ?? {}).length);
    for (const [port, value] of Object.entries(status ?? {})) {
      expect(operation.outputs[port]).toEqual({value, format: 'uint32', length: 1});
    }
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const views: Record<string, GraphDataView> = {};
    for (const [port, shape] of Object.entries(shapes)) {
      views[operation.inputs[port]] = createTransientView(graph, port, shape.format, shape.length);
    }
    for (const [port, shape] of Object.entries(operation.outputs)) {
      views[shape.value] = createTransientView(graph, port, shape.format, shape.length);
    }
    const resolved: string[] = [];
    let lowered = false;
    const context = {
      resolveVector(value: string) {
        resolved.push(value);
        return {data: [views[value]]};
      },
      lower(contributor: GPUTerrainCatchmentSummary) {
        lowered = true;
        expect(contributor).toBeInstanceOf(GPUTerrainCatchmentSummary);
        expect(Object.keys(contributor.props.output.status ?? {})).toEqual(
          Object.keys(status ?? {})
        );
        for (const [port, name] of Object.entries(status ?? {})) {
          expect(
            contributor.props.output.status?.[
              port as 'converged' | 'invalidCount' | 'iterationLimitReached'
            ]
          ).toBe(views[name]);
        }
      }
    } as unknown as Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1];
    try {
      GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(operation, context);
      expect(lowered).toBe(true);
      expect(resolved).toEqual(expect.arrayContaining(Object.values(status ?? {})));
    } finally {
      device.destroy();
    }
    const cpu = new CPUTerrainCatchmentSummaryOperation(props);
    const outputs = cpu.contract.outputs;
    if (!outputs || typeof outputs === 'function')
      throw new Error('Expected static catchment CPU output declarations');
    expect(Object.keys(outputs)).toHaveLength(Object.keys(operation.outputs).length);
  }
  for (const status of [
    {converged: ''},
    {converged: 'elevation'},
    {converged: 'profiles'},
    {converged: 'same', invalidCount: 'same'}
  ]) {
    for (const Operation of [
      GPUTerrainCatchmentSummaryOperation,
      CPUTerrainCatchmentSummaryOperation
    ]) {
      expect(() => new Operation({...base, output: {...base.output, status}})).toThrow(
        /^catchment-test/
      );
    }
  }
});

it('pins four-cell aggregate profile diagnostics against hand-counted statuses', () => {
  for (const scene of makeCatchmentStatusCases()) {
    const result = computeTerrainCatchmentSummaryOnCPU(
      {
        elevation: Float32Array.of(4, 3, 2, 1),
        basinIds: Uint32Array.from(scene.basins),
        streams: Uint32Array.from(scene.streams),
        flowDirections: Uint32Array.from(scene.directions)
      },
      {
        ...makeCatchmentProps(),
        grid: {...makeCatchmentProps().grid, width: 4, height: 1},
        maximumSteps: scene.maximumSteps
      }
    );
    expect(
      [result.converged[0], result.invalidCount[0], result.iterationLimitReached[0]],
      scene.name
    ).toEqual(scene.expected);
  }
});

it('ignores extra keys on structurally wider status objects in both operation contracts', () => {
  const base = makeCatchmentProps();
  const status = {converged: 'ok', statistics: 'side', overflow: 'unsupported'};
  const props = {...base, output: {...base.output, status}};
  const gpu = new GPUTerrainCatchmentSummaryOperation(props);
  const cpu = new CPUTerrainCatchmentSummaryOperation(props);
  expect(Object.keys(gpu.outputs)).toEqual([
    'statistics',
    'hypsometry',
    'profiles',
    'profileStatus',
    'converged'
  ]);
  expect(gpu.outputs['statistics']).toEqual({
    value: base.output.statistics,
    format: 'float32',
    length: base.basinCount * 6
  });
  for (const contract of [gpu.contract, cpu.contract]) {
    const outputs = contract.outputs;
    if (!outputs || typeof outputs === 'function')
      throw new Error('Expected static catchment output declarations');
    expect(Object.keys(outputs)).toEqual([
      'statistics',
      'hypsometry',
      'profiles',
      'profileStatus',
      'ok'
    ]);
    expect(outputs['statistics']).toEqual({
      type: 'vector',
      format: 'float32',
      length: base.basinCount * 6
    });
  }
});

it('rejects padded aggregate catchment status readbacks', () => {
  for (const port of ['done', 'invalid', 'limited']) {
    const result: Record<string, number[]> = {done: [1], invalid: [0], limited: [0]};
    result[port].push(99);
    expect(() => expectCatchmentStatusReadbacks(result, [1, 0, 0])).toThrow();
  }
});

it('ignores unrelated published columns in a structurally wider catchment record', () => {
  const input = makeCatchmentInput();
  const record = {...input, statistics: new Float32Array(18), hypsometry: new Float32Array(12)};
  const result = computeTerrainCatchmentSummaryOnCPU(record, makeCatchmentProps());
  // Four cells per basin, each area 6: area=24; mean heights 13.5 and 7.5.
  expect(Array.from(result.statistics.subarray(0, 12))).toEqual([
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
    Math.fround(1 / 12)
  ]);
  expect(result).toEqual(computeTerrainCatchmentSummaryOnCPU(input, makeCatchmentProps()));
});

it('constructs typed CPU props with individually optional named status destinations', () => {
  const base = makeCatchmentProps();
  for (const status of [
    undefined,
    {converged: 'done'},
    {invalidCount: 'invalid'},
    {iterationLimitReached: 'limited'}
  ]) {
    const props: CPUTerrainCatchmentSummaryOperationProps = {
      ...base,
      output: {...base.output, status}
    };
    const operation = new PublicCPUTerrainCatchmentSummaryOperation(props);
    expect(Object.keys(operation.outputs)).toHaveLength(4 + Object.keys(status ?? {}).length);
  }
});

it('review excluded cells take precedence over unknown stream membership', () => {
  const base = makeCatchmentProps();
  const options = {
    ...base,
    grid: {...base.grid, width: 1, height: 1},
    basinCount: 1,
    histogramBins: 1
  };
  for (const [validity, basin] of [
    [0, 0],
    [1, 2],
    [0, 2]
  ]) {
    const result = computeTerrainCatchmentSummaryOnCPU(
      {
        elevation: Float32Array.of(1),
        validity: Uint32Array.of(validity),
        basinIds: Uint32Array.of(basin),
        streams: Uint32Array.of(0xffffffff),
        flowDirections: Uint32Array.of(0)
      },
      options
    );
    // The sole row is excluded by validity or basin domain, so no eligible profile can fail.
    expect(Array.from(result.profileStatus)).toEqual([0]);
    expect(Array.from(result.converged)).toEqual([1]);
    expect(Array.from(result.invalidCount)).toEqual([1]);
    expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  }
});
