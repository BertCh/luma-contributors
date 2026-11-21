// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULineChunk,
  getGPULineChunkParameterValues
} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import {
  GPULineSegmentize,
  getGPULineSegmentizeParameterValues
} from '../../../src/gpu-spatial-analysis/line-segmentize/index';
import {
  GPULineSmooth,
  getGPULineSmoothParameterValues
} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-smooth';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createLineLineFixture} from './line-test-utils';

function createMovingEndpointLine(): Float32Array {
  const positions = new Float32Array(193 * 2);
  for (let row = 1; row < 193; row++) {
    positions[row * 2] = Math.min(row, 64) * 16;
    positions[row * 2 + 1] = Math.max(row - 64, 0) * 2 ** -20;
  }
  return positions;
}

// Independent exact binary sum: 64*16 + 128*2^-20 = 1024 + 2^-13.
// Endpoint is (1024,2^-13); no stationary row exposes a delayed carry.
for (const route of ['chunk', 'segmentize', 'smooth'] as const) {
  it(`W5-11 ${route} retains the moving endpoint and its compensated measure`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createLineLineFixture(device, {
      inputs: {
        positions: {format: 'float32x2', values: createMovingEndpointLine()},
        lineOffsets: {format: 'uint32', values: Uint32Array.of(0, 193)}
      },
      capacity: 512,
      lineCapacity: route === 'chunk' ? 2 : 1,
      createContributor: (inputs, parameters, output) => {
        const geometry = {
          positions: inputs['positions'] as GraphDataView<'float32x2'>,
          lineOffsets: inputs['lineOffsets'] as GraphDataView<'uint32'>,
          parameters,
          output
        };
        if (route === 'chunk') return new GPULineChunk({...geometry, mode: 'chunk'});
        if (route === 'smooth') return new GPULineSmooth({...geometry, iterationCount: 1});
        return new GPULineSegmentize(geometry);
      }
    });
    try {
      const parameters =
        route === 'chunk'
          ? getGPULineChunkParameterValues({chunkLength: 1024})
          : route === 'smooth'
            ? getGPULineSmoothParameterValues()
            : getGPULineSegmentizeParameterValues({maximumSegmentLength: 16});
      const result = await fixture.run(parameters);
      expect(result.count).toBe(route === 'chunk' ? 2 : 1);
      expect(result.requiredCount).toBe(route === 'chunk' ? 2 : 1);
      expect(result.overflow).toBe(0);
      if (route === 'chunk') {
        const cut = result.lineOffsets[1];
        expect(result.positions[cut - 1]).toEqual([1024, 0]);
        expect(result.positions[cut]).toEqual([1024, 0]);
      }
      const terminal = result.lineOffsets[result.count] - 1;
      expect(result.positions[terminal].every(Number.isFinite)).toBe(true);
      expect(result.positions[terminal]).toEqual([1024, 0.0001220703125]);
      expect(Number.isFinite(result.measures[terminal])).toBe(true);
      // Chunk metadata stays in the original source line measure domain.
      expect(result.measures[terminal]).toBe(1024.0001220703125);
    } finally {
      fixture.destroy();
    }
  });
}

it('B8 terminal zero-width substring emits two identical known endpoints', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createLineLineFixture(device, {
    inputs: {
      positions: {format: 'float32x2', values: createMovingEndpointLine()},
      lineOffsets: {format: 'uint32', values: Uint32Array.of(0, 193)}
    },
    capacity: 2,
    lineCapacity: 1,
    createContributor: (inputs, parameters, output) =>
      new GPULineChunk({
        positions: inputs['positions'] as GraphDataView<'float32x2'>,
        lineOffsets: inputs['lineOffsets'] as GraphDataView<'uint32'>,
        parameters,
        output,
        mode: 'substring'
      })
  });
  try {
    const result = await fixture.run(
      getGPULineChunkParameterValues({
        startMeasure: 1024.0001220703125,
        endMeasure: 1024.0001220703125
      })
    );
    expect(result.count).toBe(1);
    expect(result.overflow).toBe(0);
    expect(result.lineOffsets.slice(0, 2)).toEqual([0, 2]);
    expect(result.positions).toEqual([
      [1024, 0.0001220703125],
      [1024, 0.0001220703125]
    ]);
    expect(result.measures).toEqual([1024.0001220703125, 1024.0001220703125]);
  } finally {
    fixture.destroy();
  }
});
