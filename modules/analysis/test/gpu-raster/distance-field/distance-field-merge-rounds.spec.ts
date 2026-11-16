// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('retains exact-distance candidates through shrinking merge rounds and an unpaired final band', async () => {
  const {GPUDistanceField} = await import(
    '../../../src/raster-analysis/distance-field/gpu-distance-field'
  );
  const {getGPUDistanceFieldParameterValues} = await import(
    '../../../src/raster-analysis/distance-field/distance-field-parameters'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 193;
  const height = 3;
  const length = width * height;
  const grid = {
    width,
    height,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const maskValues = new Uint32Array(length);
  maskValues[0] = 1;
  maskValues[length - 1] = 3;
  const mask = createInputBuffer(device, maskValues);
  const settings = createInputBuffer(device, getGPUDistanceFieldParameterValues({}, grid));
  const distances = createOutputBuffer(device, length);
  const allocation = createOutputBuffer(device, length);
  const graph = new GPUCommandGraph(device);
  graph.add(
    new GPUDistanceField({
      grid,
      parameters: importGraphBuffer(graph, 'settings', settings, 'float32', 8),
      seedMask: importGraphBuffer(graph, 'mask', mask, 'uint32', length),
      output: {
        distances: importGraphBuffer(graph, 'distances', distances, 'float32', length),
        allocation: importGraphBuffer(graph, 'allocation', allocation, 'uint32', length)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(distances, length);
    const ids = await readUint32(allocation, length);
    for (let cell = 0; cell < length; cell++) {
      const column = cell % width;
      const row = Math.floor(cell / width);
      const first = Math.hypot(column, row);
      const second = Math.hypot(width - 1 - column, height - 1 - row);
      const expected = Math.min(first, second);
      expect(Math.abs(actual[cell] - expected)).toBeLessThanOrEqual(
        Math.max(1e-6, expected * 2e-7)
      );
      expect(ids[cell]).toBe(first <= second ? 0 : 2);
    }
  } finally {
    compiled.destroy();
    for (const buffer of [mask, settings, distances, allocation]) buffer.destroy();
  }
});
