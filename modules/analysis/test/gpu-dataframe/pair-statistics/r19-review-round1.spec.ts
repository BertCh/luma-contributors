// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Buffer} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPURipleyDistanceFunctions} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley-distance-functions';
import {getGPURipleyDistanceFunctionsParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-distance-parameters';
import {GPUPointPatternIndices} from '../../../src/dataframe-analysis/pair-statistics/gpu-point-pattern-indices';
import {getGPUPointPatternIndicesParameterValues} from '../../../src/dataframe-analysis/pair-statistics/point-pattern-indices-parameters';
import {GPUSpaceTimeKFunction} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-k-function';
import {getGPUSpaceTimeKFunctionParameterValues} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-nodes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createPairStatisticsHarness,
  createPairStatisticsWindowHarness
} from './pair-statistics-harness';

it('B1 exact Hanisch G retains uncensored unresolved events at a large origin', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createPairStatisticsHarness(device, {
    positions: Float32Array.from([
      -1 / 8,
      15 / 128,
      -1 / 64,
      15 / 128,
      3,
      15 / 128,
      3 + 1 / 256,
      15 / 128
    ]),
    parameterLength: 8,
    outputs: {g: {format: 'float32', length: 1}},
    createContributor: views =>
      new GPURipleyDistanceFunctions({
        positions: views.positions,
        parameters: views.parameters,
        gridSize: [256, 1],
        referenceGrid: [1, 1],
        radiusCount: 1,
        hanischSearch: 'exact',
        output: {g: views.outputs.g as never}
      })
  });
  try {
    const result = await harness.run(
      getGPURipleyDistanceFunctionsParameterValues({
        bounds: [-1048576, 0, 1048576, 1024],
        maximumDistance: 1 / 128,
        edgeCorrection: 'hanisch'
      })
    );
    // Two d=7/64 events beyond reported radius remain uncensored; two d=1/256 events
    // contribute to the numerator. Integer Hanisch weights are respectively 16388 and 16384.
    expect(result.g).toHaveLength(1);
    expect(Number.isFinite(result.g[0])).toBe(true);
    expect(result.g[0]).toBeCloseTo(0.4999389722934212, 6);
  } finally {
    harness.destroy();
  }
});

it('B5 censored F product-limit survival differs from the empirical CDF', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createPairStatisticsHarness(device, {
    positions: Float32Array.from([1, 3]),
    parameterLength: 8,
    outputs: {f: {format: 'float32', length: 2}},
    createContributor: views =>
      new GPURipleyDistanceFunctions({
        positions: views.positions,
        parameters: views.parameters,
        gridSize: [3, 3],
        referenceGrid: [3, 1],
        radiusCount: 2,
        output: {f: views.outputs.f as never}
      })
  });
  try {
    // References (1,3),(3,3),(5,3) have (d,b)=(0,1),(2,3),(4,1).
    // Risk set 3 loses an event and a censoring in slot 0; remaining risk 1 has an event at 2.
    for (const [mode, expected] of [
      ['none', 2 / 3],
      ['kaplan-meier', 1]
    ] as const) {
      const result = await harness.run(
        getGPURipleyDistanceFunctionsParameterValues({
          bounds: [0, 0, 6, 6],
          maximumDistance: 2,
          edgeCorrection: mode
        })
      );
      expect(result.f).toHaveLength(2);
      expect(result.f.every(Number.isFinite)).toBe(true);
      expect(result.f[0]).toBeCloseTo(1 / 3, 6);
      expect(result.f[1]).toBeCloseTo(expected, 6);
    }
  } finally {
    harness.destroy();
  }
});

it('quadrat product ties keep the edge predecessor in the lower quadrat', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createPairStatisticsHarness(device, {
    positions: Float32Array.from([2396754.75, 0.5, 2396755, 0.5]),
    parameterLength: 8,
    outputs: {counts: {format: 'uint32', length: 7}},
    createContributor: views =>
      new GPUPointPatternIndices({
        positions: views.positions,
        parameters: views.parameters,
        gridSize: [4, 1],
        quadratGrid: [7, 1],
        output: {quadratCounts: views.outputs.counts as never}
      })
  });
  try {
    const result = await harness.run(
      getGPUPointPatternIndicesParameterValues({
        bounds: [0, 0, 3355457, 1],
        indexCellSizeHint: 1,
        clarkEvansCorrection: 'none'
      })
    );
    expect(result.counts).toEqual([0, 0, 0, 0, 1, 1, 0]);
  } finally {
    harness.destroy();
  }
});

it('two occupied unsampled quadrats invalidate through a single statistics writer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const ring = [0, 0, 1000, 0, 1000, 400, 3000, 400, 3000, 405, 1000, 405, 1000, 1000, 0, 1000];
  const harness = createPairStatisticsWindowHarness(device, {
    positions: Float32Array.from([100, 100, 1500, 402, 2500, 402]),
    parameterLength: 8,
    window: {
      positions: Float64Array.from(ring),
      ringOffsets: Uint32Array.from([0, 8]),
      polygonOffsets: Uint32Array.from([0, 1]),
      featureOffsets: Uint32Array.from([0, 1])
    },
    outputs: {statistics: {format: 'float32', length: 7}, counts: {format: 'uint32', length: 6}},
    createContributor: (views, studyWindow) =>
      new GPUPointPatternIndices({
        positions: views.positions,
        parameters: views.parameters,
        studyWindow,
        gridSize: [4, 4],
        quadratGrid: [3, 2],
        quadratSamples: 16,
        output: {
          quadratStatistics: views.outputs.statistics as never,
          quadratCounts: views.outputs.counts as never
        }
      })
  });
  try {
    const result = await harness.run(
      getGPUPointPatternIndicesParameterValues({
        bounds: 'study-window',
        indexCellSizeHint: 1,
        clarkEvansCorrection: 'none'
      })
    );
    expect(result.counts).toEqual([1, 1, 1, 0, 0, 0]);
    expect(result.statistics).toHaveLength(7);
    for (const index of [1, 2, 3, 4, 6]) expect(result.statistics[index]).toBeNaN();
  } finally {
    harness.destroy();
  }
});

it('space-time PCF retains small positive Epanechnikov products and reports a clamped K scale', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const clamped of [false, true]) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    let serial = 0;
    const input = <Format extends 'float32' | 'float32x2'>(format: Format, data: number[]) => {
      const buffer = createInputBuffer(device, Float32Array.from(data));
      buffers.push(buffer);
      return importGraphBuffer(
        graph,
        `input-${serial++}`,
        buffer,
        format,
        data.length / (format === 'float32x2' ? 2 : 1)
      );
    };
    const output = (format: 'float32' | 'uint32') => {
      const buffer = createOutputBuffer(device, 1);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, 1)};
    };
    const k = output('float32');
    const correlation = output('float32');
    const status = output('uint32');
    const separation = 2 - 1 / 1024;
    graph.add(
      new GPUSpaceTimeKFunction({
        positions: input('float32x2', [1, 1, clamped ? 1.5 : 1 + separation, 1]),
        times: input('float32', [1, clamped ? 1.5 : 1 + separation]),
        intensities: input('float32', clamped ? [32, 32] : [2 / 1048576, 2 / 1048576]),
        parameters: input(
          'float32',
          Array.from(
            getGPUSpaceTimeKFunctionParameterValues({
              bounds: clamped ? [0, 0, 65536, 65536] : [0, 0, 32, 32],
              timeWindow: clamped ? [0, 65536] : [0, 1024],
              maximumDistance: 1,
              maximumTimeLag: 1,
              spatialBandwidth: 1,
              temporalBandwidth: 1
            })
          )
        ),
        gridSize: [4, 4],
        distanceClassCount: 1,
        timeClassCount: 1,
        edgeCorrection: 'none',
        pairCorrelationKernel: 'epanechnikov',
        output: {
          k: k.view as never,
          pairCorrelation: correlation.view as never,
          status: status.view as never
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      if (clamped) {
        expect(await readUint32(status.buffer, 1)).toEqual([3]);
        expect((await readFloat32(k.buffer, 1))[0]).toBeNaN();
        expect((await readFloat32(correlation.buffer, 1))[0]).toBeNaN();
      } else {
        const actual = await readFloat32(correlation.buffer, 1);
        expect(actual).toHaveLength(1);
        expect(Number.isFinite(actual[0])).toBe(true);
        expect(actual[0]).toBeCloseTo(0.08943725041215612, 6);
        expect(await readUint32(status.buffer, 1)).toEqual([0]);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});
