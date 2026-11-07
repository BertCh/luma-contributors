// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

function normalizeCharts(chartPoints: number[][]): number[] {
  return chartPoints.flatMap(point => {
    const length = Math.hypot(...point);
    return point.map(value => value / length);
  });
}

it('spherical polygon contains validates each ring once and rejects self-crossing rings per query', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const square = [
    [1, -0.3, -0.3],
    [1, 0.3, -0.3],
    [1, 0.3, 0.3],
    [1, -0.3, 0.3]
  ];
  // Bow-tie: edges 0-1 and 2-3 cross.
  const bowTie = [
    [1, -0.3, -0.3],
    [1, 0.3, 0.3],
    [1, 0.3, -0.3],
    [1, -0.3, 0.3]
  ];
  const positions = normalizeCharts([...square, ...bowTie]);
  const queries = normalizeCharts([
    [1, 0, 0],
    [1, 0.9, 0.9],
    [1, 0, 0]
  ]);
  const buffers = [
    createInputBuffer(device, new Float32Array(positions)),
    createInputBuffer(device, new Uint32Array([0, 4, 8])),
    createInputBuffer(device, new Uint32Array([0, 1, 2])),
    createInputBuffer(device, new Float32Array(queries)),
    createInputBuffer(device, new Uint32Array([0, 0, 1])),
    createOutputBuffer(device, 3),
    createOutputBuffer(device, 3)
  ];
  const view = <Format extends 'float32x3' | 'uint32'>(
    index: number,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, `buffer-${index}`, buffers[index], format, length);
  graph.add(
    new GPUSphericalPolygonContains({
      positions: view(0, 'float32x3', 8),
      ringOffsets: view(1, 'uint32', 3),
      polygonOffsets: view(2, 'uint32', 3),
      queries: view(3, 'float32x3', 3),
      polygonIndices: view(4, 'uint32', 3),
      hemisphereCenter: [1, 0, 0],
      output: {contains: view(5, 'uint32', 3), status: view(6, 'uint32', 3)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(buffers[6], 3)).toEqual([0, 0, 1]);
    expect(await readUint32(buffers[5], 3)).toEqual([1, 0, 0]);
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
