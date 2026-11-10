// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUSpaceTimeDiggleRandomParameterValues,
  getGPUSpaceTimeKFunctionParameterValues,
  GPU_SPACE_TIME_DIGGLE_MAXIMUM_PERMUTATIONS,
  GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH,
  GPUSpaceTimeDiggleTest,
  GPUSpaceTimeKFunction
} from '../../../src/dataframe-analysis/pair-statistics/index';

const SETTINGS = {
  bounds: [0, 0, 10, 8],
  maximumDistance: 4,
  maximumTimeLag: 6,
  timeWindow: [0, 20]
} as const;

it('packs the parameter layout and rejects invalid settings', () => {
  expect(Array.from(getGPUSpaceTimeKFunctionParameterValues(SETTINGS))).toEqual([
    0, 0, 10, 8, 4, 4, 6, 0, 20, 0, 0
  ]);
  // The pair search radius grows by the spatial bandwidth, for the pair correlation.
  expect(
    Array.from(
      getGPUSpaceTimeKFunctionParameterValues({
        ...SETTINGS,
        spatialBandwidth: 0.5,
        temporalBandwidth: 1.5
      })
    )
  ).toEqual([0, 0, 10, 8, 4.5, 4, 6, 0, 20, 0.5, 1.5]);
  for (const bad of [
    {...SETTINGS, bounds: [0, 0, 0, 8] as const},
    {...SETTINGS, bounds: [0, 0, Number.NaN, 8] as const},
    {...SETTINGS, maximumDistance: 0},
    {...SETTINGS, maximumTimeLag: -1},
    {...SETTINGS, timeWindow: [5, 5] as const},
    {...SETTINGS, spatialBandwidth: -1},
    {...SETTINGS, temporalBandwidth: Number.POSITIVE_INFINITY}
  ]) {
    expect(() => getGPUSpaceTimeKFunctionParameterValues(bad)).toThrow(
      /getGPUSpaceTimeKFunctionParameterValues/
    );
  }
  expect(() => getGPUSpaceTimeKFunctionParameterValues(SETTINGS, new Float32Array(10))).toThrow(
    /target must hold 11/
  );
  expect(Array.from(getGPUSpaceTimeDiggleRandomParameterValues(2 ** 40 + 3))).toEqual([3, 256]);
  expect(() => getGPUSpaceTimeDiggleRandomParameterValues(-1)).toThrow(/seed/);
  expect(() => getGPUSpaceTimeDiggleRandomParameterValues(1.5)).toThrow(/seed/);
});

function createViews(rows: number, overrides: Record<string, unknown> = {}) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  return {
    device,
    graph,
    view,
    props: {
      positions: view('positions', 'float32x2', rows),
      times: view('times', 'float32', rows),
      parameters: view('parameters', 'float32', 11),
      gridSize: [8, 8] as const,
      distanceClassCount: 4,
      timeClassCount: 3,
      ...overrides
    }
  };
}

it('GPUSpaceTimeKFunction validates props at construction with the ID and the prop', () => {
  const {view, props} = createViews(20);
  const output = {k: view('k', 'float32', 12)};
  expect(() => new GPUSpaceTimeKFunction({...props, output})).not.toThrow();
  const build =
    (overrides: Record<string, unknown>, outputOverrides: Record<string, unknown> = {}) =>
    () =>
      new GPUSpaceTimeKFunction({
        ...props,
        ...overrides,
        output: {...output, ...outputOverrides}
      } as never);
  expect(build({distanceClassCount: 0})).toThrow(/space-time-k-function distanceClassCount/);
  expect(build({timeClassCount: 65})).toThrow(/timeClassCount must be an integer in \[1, 64\]/);
  expect(build({edgeCorrection: 'translate'})).toThrow(/edgeCorrection/);
  expect(build({pairCorrelationKernel: 'gaussian'})).toThrow(/pairCorrelationKernel/);
  expect(build({}, {k: view('short-k', 'float32', 11)})).toThrow(/output.k must hold 12 rows/);
  expect(build({}, {pairCounts: view('counts', 'float32', 12)})).toThrow(/output.pairCounts/);
  expect(build({times: view('short-times', 'float32', 19)})).toThrow(/times length must equal/);
  expect(
    build({intensities: view('intensities', 'float32', 20)}, {d: view('d', 'float32', 12)})
  ).toThrow(/homogeneous estimator only/);
  expect(build({parameters: view('short-parameters', 'float32', 10)})).toThrow(
    /parameters must hold 11/
  );
  expect(build({gridSize: [0, 8]})).toThrow(/gridSize/);
  // 64 x 64 classes cannot be held by the pair histogram accumulators.
  expect(
    build({distanceClassCount: 64, timeClassCount: 64}, {k: view('big-k', 'float32', 4096)})
  ).toThrow(/at most 2048 histogram accumulators/);
  // Local statistics are limited to 64 classes per event.
  expect(
    build(
      {distanceClassCount: 9, timeClassCount: 8},
      {k: view('k72', 'float32', 72), local: view('local', 'float32', 20 * 72)}
    )
  ).toThrow(/output.local needs distanceClassCount \* timeClassCount <= 64/);
  // Outputs must not share a buffer with each other or with an input.
  expect(build({}, {pairCounts: output.k as never})).toThrow();
  expect(build({spatialContext: {coordinateSystem: 'longitude-latitude'}} as never)).toThrow();
});

it('GPUSpaceTimeKFunction emits audited kernels within eight storage bindings', () => {
  for (const edgeCorrection of ['none', 'isotropic'] as const) {
    for (const [name, outputs, intensities] of [
      ['plain', {}, false],
      ['marginals', {marginals: true}, false],
      ['pair correlation', {pairCorrelation: true}, true],
      ['local', {local: true}, false],
      ['everything', {marginals: true, pairCorrelation: true, local: true}, false]
    ] as const) {
      const rows = 40;
      const {device, graph, view, props} = createViews(rows, {edgeCorrection});
      const bins = 12;
      const output: Record<string, unknown> = {
        k: view('k', 'float32', bins),
        pairCounts: view('counts', 'uint32', bins),
        status: view('status', 'uint32', 1)
      };
      if ('marginals' in outputs) {
        output['spatialK'] = view('spatial-k', 'float32', 4);
        output['temporalK'] = view('temporal-k', 'float32', 3);
        output['d'] = view('d', 'float32', bins);
      }
      if ('pairCorrelation' in outputs) {
        output['pairCorrelation'] = view('pcf', 'float32', bins);
      }
      if ('local' in outputs) {
        output['local'] = view('local', 'float32', rows * bins);
      }
      const nodes = new GPUSpaceTimeKFunction({
        ...props,
        edgeCorrection,
        intensities: intensities ? view('intensities', 'float32', rows) : undefined,
        output
      } as never).getCommandNodes(graph);
      const sources = getGPUCommandNodeKernelSources(nodes);
      expect(sources.length, name).toBeGreaterThan(5);
      for (const {source, constants} of sources) {
        expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
        expect(auditWGSLModule(source, constants), name).toEqual([]);
      }
      device.destroy();
    }
  }
});

function createDiggle(
  rows: number,
  overrides: Partial<{
    distanceClassCount: number;
    timeClassCount: number;
    permutationCount: number;
    edgeCorrection: 'none' | 'isotropic';
  }> = {}
) {
  const {device, graph, view, props} = createViews(rows, overrides);
  const permutationCount = overrides.permutationCount ?? 19;
  const bins = (overrides.distanceClassCount ?? 4) * (overrides.timeClassCount ?? 3);
  const test = new GPUSpaceTimeDiggleTest({
    ...props,
    randomParameters: view('random', 'uint32', 2),
    permutationCount,
    output: {
      d: view('d', 'float32', bins),
      statistics: view('statistics', 'float32', permutationCount + 1),
      summary: view('summary', 'float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH)
    }
  } as never);
  return {device, graph, test};
}

it('GPUSpaceTimeDiggleTest validates props at construction', () => {
  expect(() => createDiggle(30)).not.toThrow();
  expect(() => createDiggle(30, {permutationCount: 0})).toThrow(
    /space-time-diggle-test permutationCount must be an integer in \[1, 999\]/
  );
  expect(() =>
    createDiggle(30, {permutationCount: GPU_SPACE_TIME_DIGGLE_MAXIMUM_PERMUTATIONS + 1})
  ).toThrow(/permutationCount/);
  expect(() => createDiggle(30, {edgeCorrection: 'translate' as never})).toThrow(/edgeCorrection/);
  // 64 x 64 classes: two accumulator channels per slot do not fit.
  expect(() => createDiggle(30, {distanceClassCount: 64, timeClassCount: 64})).toThrow(/too large/);
  const {view, props} = createViews(10);
  expect(
    () =>
      new GPUSpaceTimeDiggleTest({
        ...props,
        randomParameters: view('random', 'uint32', 1),
        permutationCount: 9,
        output: {
          d: view('d', 'float32', 12),
          statistics: view('statistics', 'float32', 10),
          summary: view('summary', 'float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH)
        }
      } as never)
  ).toThrow(/randomParameters must hold 2/);
  expect(
    () =>
      new GPUSpaceTimeDiggleTest({
        ...props,
        randomParameters: view('random2', 'uint32', 2),
        permutationCount: 9,
        output: {
          d: view('d2', 'float32', 12),
          statistics: view('statistics2', 'float32', 9),
          summary: view('summary2', 'float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH)
        }
      } as never)
  ).toThrow(/output.statistics must hold 10 rows/);
});

it('GPUSpaceTimeDiggleTest never emits more command nodes than it estimates', () => {
  for (const [rows, shape] of [
    [30, {}],
    [30, {permutationCount: 999}],
    [30, {distanceClassCount: 8, timeClassCount: 16, permutationCount: 999}],
    [100_000, {distanceClassCount: 8, timeClassCount: 8, permutationCount: 999}],
    [8_388_608, {distanceClassCount: 4, timeClassCount: 4, permutationCount: 999}]
  ] as const) {
    const {device, graph, test} = createDiggle(rows, shape);
    const nodes = test.getCommandNodes(graph);
    expect(nodes.length, `${rows} rows ${JSON.stringify(shape)}`).toBeLessThanOrEqual(
      test.estimatedCommandNodeCount
    );
    expect(nodes.length).toBeGreaterThan(20);
    device.destroy();
  }
});

it('GPUSpaceTimeDiggleTest emits audited kernels within eight storage bindings', () => {
  for (const edgeCorrection of ['none', 'isotropic'] as const) {
    const {device, graph, test} = createDiggle(40, {edgeCorrection, permutationCount: 99});
    const sources = getGPUCommandNodeKernelSources(test.getCommandNodes(graph));
    expect(sources.length).toBeGreaterThan(10);
    for (const {source, constants} of sources) {
      expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(source, constants)).toEqual([]);
    }
    device.destroy();
  }
});
