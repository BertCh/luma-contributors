// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUAdaptiveBandwidthKernelDensity,
  getGPUAdaptiveBandwidthKernelDensityParameterValues
} from '../../../src/gpu-spatial-analysis/point-density/gpu-adaptive-bandwidth-kernel-density';
import {GPUSpaceTimeKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-kernel-density';
import {GPUSpaceTimeRelativeRisk} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-relative-risk';
import {getGPUSpaceTimeKernelDensityParameterValues} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-math';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

for (const kind of ['adaptive', 'space-time', 'risk'] as const) {
  it(`${kind} writes independent Gaussian expectations to packed fields of one imported output buffer`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
    const buffers: Buffer[] = [];
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const input = (name: string, values: number[]) =>
      importGraphBuffer(
        graph,
        name,
        track(createInputBuffer(device, Float32Array.from(values))),
        'float32',
        values.length
      );
    const positions = importGraphBuffer(
      graph,
      'positions',
      track(createInputBuffer(device, new Float32Array([0.5, 0.5, 0.5, 0.5]))),
      'float32x2',
      2
    );
    const times = input('times', [0, 0]);
    const weights = input('weights', kind === 'adaptive' ? [1, 3] : [2, Number.NaN]);
    const parameterValues =
      kind === 'adaptive'
        ? getGPUAdaptiveBandwidthKernelDensityParameterValues({bandwidth: 1}, grid)
        : getGPUSpaceTimeKernelDensityParameterValues(
            {bandwidth: 1, temporalBandwidth: 1, firstSliceTime: 0, sliceInterval: 1},
            grid,
            1
          );
    const parameters = input('parameters', Array.from(parameterValues));
    const packed = track(createOutputBuffer(device, 32));
    // One physical import, with logical output fields whose aligned binding ranges overlap.
    const handle = graph.importBuffer(
      {id: 'packed', byteLength: packed.byteLength, usage: packed.usage},
      packed
    );
    const float = (byteOffset: number, length = 1) =>
      graph.createDataView(handle, {format: 'float32', byteOffset, length});
    const integer = (byteOffset: number) =>
      graph.createDataView(handle, {format: 'uint32', byteOffset, length: 1});
    const common = {
      positions,
      weights,
      parameters,
      grid,
      indexGridSize: [1, 1] as const,
      indexBounds: [0, 0, 1, 1] as const
    };
    if (kind === 'adaptive') {
      graph.add(
        new GPUAdaptiveBandwidthKernelDensity({
          ...common,
          output: {
            values: float(0),
            weightSum: float(4),
            pointBandwidths: float(8, 2),
            pilotDensities: float(16, 2)
          }
        })
      );
    } else if (kind === 'space-time') {
      graph.add(
        new GPUSpaceTimeKernelDensity({
          ...common,
          times,
          sliceCount: 1,
          spatialEdgeCorrection: 'none',
          temporalEdgeCorrection: 'none',
          normalization: 'probability',
          output: {values: float(0), weightSum: float(4), status: {invalidCount: integer(8)}}
        })
      );
    } else {
      graph.add(
        new GPUSpaceTimeRelativeRisk({
          ...common,
          casePositions: positions,
          caseTimes: times,
          caseWeights: weights,
          controlPositions: positions,
          controlTimes: times,
          controlWeights: input('control-weights', [1, 3]),
          sliceCount: 1,
          spatialEdgeCorrection: 'none',
          temporalEdgeCorrection: 'none',
          output: {
            values: float(0),
            caseDensity: float(4),
            controlDensity: float(8),
            status: {caseInvalidCount: integer(12), controlInvalidCount: integer(16)}
          }
        })
      );
    }
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(packed, 6);
      const integers = await readUint32(packed, 5);
      if (kind === 'adaptive') {
        // Identical locations: pilot = sum(weights) / (2 pi); geometric mean equals that pilot,
        // so the Abramson factor is one and every point bandwidth is exactly h = 1.
        expect(actual[0]).toBeCloseTo(4 / (2 * Math.PI), 5);
        expect(actual[1]).toBe(4);
        for (const bandwidth of actual.slice(2, 4)) expect(bandwidth).toBeCloseTo(1, 6);
        for (const pilot of actual.slice(4, 6)) expect(pilot).toBeCloseTo(4 / (2 * Math.PI), 5);
      } else if (kind === 'space-time') {
        expect(actual[0]).toBeCloseTo((2 * Math.PI) ** -1.5, 5);
        expect(actual[1]).toBe(2);
        expect(integers[2]).toBe(1);
      } else {
        expect(actual[0]).toBeCloseTo(0, 6);
        expect(actual[1]).toBeCloseTo((2 * Math.PI) ** -1.5, 5);
        expect(actual[2]).toBeCloseTo((2 * Math.PI) ** -1.5, 5);
        expect(integers.slice(3, 5)).toEqual([1, 0]);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
