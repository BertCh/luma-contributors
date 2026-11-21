// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpaceTimeDiggleTestOperation} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test-operation';
import {CPUSpaceTimeDiggleTestOperation} from '../../../src/dataframe-analysis/pair-statistics/cpu-space-time-diggle-test-operation';
import {constructRasterOperationContributor} from '../../utils/raster-operation-test-utils';
import {makeCPUInputRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {computeSpaceTimeDiggleTestOnCPU} from '../../../src/dataframe-analysis/pair-statistics/cpu-space-time-diggle-test-operation';
import {expectLiteralRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {
  DIGGLE_INPUT,
  DIGGLE_OPTIONS,
  DIGGLE_EXPECTED,
  createDiggleOutputPoison
} from './pair-output-oracles-fixture';
import {computeCoregionalizationFitOnCPU} from '../../../src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation';
import {getGPUCoregionalizationFitParameterValues} from '../../../src/dataframe-analysis/pair-statistics/coregionalization-parameters';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import type {GPUSpaceTimeDiggleTest} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test';
import type {
  GPUSpatialContext,
  GPUTemporalContext
} from '../../../src/analysis-core/contracts/index';

it('both Diggle forms reject four positions when rowCount declares three', () => {
  const props = {
    ...DIGGLE_OPTIONS,
    id: 'diggle-shape',
    rowCount: 3,
    gridSize: [2, 2] as const,
    positions: 'positions',
    times: 'times',
    parameters: 'parameters',
    randomParameters: 'random',
    output: {d: 'd', statistics: 'statistics', summary: 'summary'}
  };
  expect(() =>
    constructRasterOperationContributor(new GPUSpaceTimeDiggleTestOperation(props), {
      positions: {format: 'float32x2', length: 4},
      times: {format: 'float32', length: 4},
      parameters: {format: 'float32', length: 11},
      randomParameters: {format: 'uint32', length: 2}
    })
  ).toThrow('diggle-shape positions length must match rowCount');
  expect(() =>
    new CPUSpaceTimeDiggleTestOperation(props).execute({
      inputs: {
        positions: makeCPUInputRows('float32x2', DIGGLE_INPUT.positions),
        times: makeCPUInputRows('float32', DIGGLE_INPUT.times),
        parameters: makeCPUInputRows('float32', DIGGLE_INPUT.parameters),
        randomParameters: makeCPUInputRows('uint32', DIGGLE_INPUT.randomParameters)
      }
    })
  ).toThrow('diggle-shape positions length must match rowCount');
  for (const rowCount of [0, -1, 1.5, NaN, Infinity])
    for (const Operation of [GPUSpaceTimeDiggleTestOperation, CPUSpaceTimeDiggleTestOperation])
      expect(() => new Operation({...props, rowCount})).toThrow(/diggle-shape rowCount/);
});

it('Diggle operations retain temporal and nested spatial contexts across repeated lowering', () => {
  const areaOfUse: [number, number, number, number] = [0, 0, 30, 30];
  const frameUnit = {name: 'meter', metersPerUnit: 1};
  const spatialContext: GPUSpatialContext = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {
      id: 'fixed-frame',
      kind: 'crs',
      origin: [10, 20],
      areaOfUse,
      unit: frameUnit
    }
  };
  const calendar = {calendarUnit: 'day' as const, interval: 1};
  const temporalContext: GPUTemporalContext = {
    unit: 'second',
    epoch: 0n,
    encoding: 'float32-relative',
    axis: {kind: 'regular', step: calendar}
  };
  const props = {
    ...DIGGLE_OPTIONS,
    id: 'context-captured',
    rowCount: 4,
    gridSize: [2, 2] as const,
    positions: 'positions',
    times: 'times',
    parameters: 'parameters',
    randomParameters: 'random',
    spatialContext,
    temporalContext,
    output: {d: 'd', statistics: 'statistics', summary: 'summary', status: 'status'}
  };
  const gpu = new GPUSpaceTimeDiggleTestOperation(props);
  const cpu = new CPUSpaceTimeDiggleTestOperation(props);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = {
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      times: createTransientView(graph, 'times', 'float32', 4),
      parameters: createTransientView(graph, 'parameters', 'float32', 11),
      randomParameters: createTransientView(graph, 'random', 'uint32', 2),
      ...Object.fromEntries(
        Object.entries(gpu.outputs).map(([name, output]) => [
          name,
          createTransientView(graph, name, output.format, output.length)
        ])
      )
    };
    const lower = () =>
      gpu.createContributor(
        port => views[port as keyof typeof views] as never
      ) as GPUSpaceTimeDiggleTest;
    lower();
    temporalContext.encoding = 'int64-words';
    expect(() => lower()).not.toThrow();
    calendar.interval = 0;
    spatialContext.frame!.origin![0] = 999;
    areaOfUse[0] = -999;
    frameUnit.metersPerUnit = 1000;
    spatialContext.frame!.kind = 'local-azimuthal-equidistant';
    spatialContext.metric = 'great-circle';
    const captured = lower();
    expect(captured.props.temporalContext).toEqual({
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'regular', step: {calendarUnit: 'day', interval: 1}}
    });
    expect(captured.props.spatialContext!.frame).toMatchObject({
      id: 'fixed-frame',
      kind: 'crs',
      origin: [10, 20],
      areaOfUse: [0, 0, 30, 30],
      unit: {metersPerUnit: 1}
    });
    const rows = cpu.execute({
      inputs: {
        positions: makeCPUInputRows('float32x2', DIGGLE_INPUT.positions),
        times: makeCPUInputRows('float32', DIGGLE_INPUT.times),
        parameters: makeCPUInputRows('float32', DIGGLE_INPUT.parameters),
        randomParameters: makeCPUInputRows('uint32', DIGGLE_INPUT.randomParameters)
      }
    }) as Record<string, Float32Array | Uint32Array>;
    expectLiteralRows(rows['status'], [0], 0);
    expectLiteralRows(rows['d'], DIGGLE_EXPECTED.d);
  } finally {
    device.destroy();
  }
});

it('direct Diggle poison rejects an unwritten zero-valued moment tail', () => {
  for (const name of ['permutationMean', 'permutationStandardDeviation'] as const) {
    const expected = DIGGLE_EXPECTED[name];
    const poison = createDiggleOutputPoison(name, expected.length);
    const actual = new Float32Array(poison.buffer);
    actual.set(expected.slice(0, -1));
    expect(() => expectLiteralRows(actual, expected, 2e-4)).toThrow();
  }
  expect(Array.from(createDiggleOutputPoison('status', 1))).toEqual([123]);
});

it('CPU Diggle ports match independent SciPy pairs and Python permutation matrices', () => {
  for (const empty of [false, true]) {
    const result = computeSpaceTimeDiggleTestOnCPU(
      {...DIGGLE_INPUT, mask: new Uint32Array(4).fill(empty ? 0 : 1)},
      DIGGLE_OPTIONS
    );
    if (!empty) {
      // Mutation guard: swapping the spatial and temporal ports must fail.
      expect(() => expectLiteralRows(result.spatialK, DIGGLE_EXPECTED.temporalK)).toThrow();
      expect(() => expectLiteralRows(result.temporalK, DIGGLE_EXPECTED.spatialK)).toThrow();
    }
    for (const name of Object.keys(DIGGLE_EXPECTED) as (keyof typeof DIGGLE_EXPECTED)[])
      expectLiteralRows(
        result[name],
        empty
          ? name === 'status'
            ? [1]
            : name === 'summary'
              ? [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0, 0]
              : DIGGLE_EXPECTED[name].map(() => NaN)
          : DIGGLE_EXPECTED[name]
      );
  }
});

it('CPU unit-lag coregionalization diagnostics count one discarded bin independently', () => {
  for (const imperfect of [false, true]) {
    // Spherical shape at h=range is 1. Diagonal matrix [2,3] exactly fits the lag matrix.
    // The second bin is unusable; residual=0, final change=0, usable=1, weight=10.
    const result = computeCoregionalizationFitOnCPU(
      {
        distances: new Float32Array([1, 2]),
        semivariances: new Float32Array(imperfect ? [2, 0, 3, 6, 0, 3] : [2, 0, 3, 2, NaN, 3]),
        pairCounts: new Uint32Array(imperfect ? [1, 3] : [10, 5]),
        parameters: getGPUCoregionalizationFitParameterValues([
          {model: 'spherical', range: 1, sill: 1}
        ])
      },
      {variableCount: 2, structureCount: 1, maximumIterations: imperfect ? 1 : 4}
    );
    // Weighted diagonals [2,3] and [6,3] fit [5,3]; SSE=(9+3)/4=3, first relative change=1.
    expectLiteralRows(result.matrices, imperfect ? [5, 0, 0, 3] : [2, 0, 0, 3]);
    expectLiteralRows(result.diagnostics, imperfect ? [3, 1, 2, 4] : [0, 0, 1, 10]);
    expectLiteralRows(result.invalidCount, [imperfect ? 0 : 1], 0);
  }
});
