// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVoronoiDiagram} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function runVoronoi(bounds: number[]) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'voronoi-clip-test'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'float32x4' | 'uint32x3' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const output = <Format extends 'float32x4' | 'uint32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const components = format === 'float32x4' ? 4 : format === 'uint32x2' ? 2 : 1;
    const buffer = createOutputBuffer(device, components * length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const segments = output('segments', 'float32x4', 8);
  const siteIds = output('site-ids', 'uint32x2', 8);
  const count = output('count', 'uint32', 1);
  const requiredCount = output('required-count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  graph.add(
    new GPUVoronoiDiagram({
      positions: input(
        'positions',
        'float32x2',
        Float32Array.from([0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5]),
        5
      ),
      triangles: input(
        'triangles',
        'uint32x3',
        Uint32Array.from([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4]),
        4
      ),
      triangleCount: input('triangle-count', 'uint32', Uint32Array.of(4), 1),
      clipBounds: input('clip-bounds', 'float32x4', Float32Array.from(bounds), 1),
      output: {
        segments: segments.view,
        siteIds: siteIds.view,
        status: {
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const written = (await readUint32(count.buffer, 1))[0];
  const result = {
    written,
    required: (await readUint32(requiredCount.buffer, 1))[0],
    segments: await readFloat32(segments.buffer, 4 * written)
  };
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
  return result;
}

it('GPUVoronoiDiagram clips internal and hull segments to the clip bounds', async () => {
  // The circumcenters of the four hull triangles lie at (0.5, 0), (1, 0.5), (0.5, 1) and (0, 0.5);
  // two of them are outside this box, so unclipped segments would escape it.
  const bounds = [0.1, 0.1, 0.75, 0.75];
  const result = await runVoronoi(bounds);
  expect(result.written).toBe(result.required);
  expect(result.written).toBeGreaterThan(0);
  const epsilon = 1e-6;
  for (let row = 0; row < result.written; row++) {
    for (const [x, y] of [
      [result.segments[4 * row], result.segments[4 * row + 1]],
      [result.segments[4 * row + 2], result.segments[4 * row + 3]]
    ]) {
      expect(x).toBeGreaterThanOrEqual(bounds[0] - epsilon);
      expect(x).toBeLessThanOrEqual(bounds[2] + epsilon);
      expect(y).toBeGreaterThanOrEqual(bounds[1] - epsilon);
      expect(y).toBeLessThanOrEqual(bounds[3] + epsilon);
    }
  }
});

it('GPUVoronoiDiagram drops segments that miss the clip bounds', async () => {
  const result = await runVoronoi([5, 5, 6, 6]);
  expect(result.written).toBe(0);
  expect(result.required).toBe(0);
});
