// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVoronoiDiagram} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function runVoronoi(
  bounds: number[],
  positionsValues = [0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5],
  triangleValues = [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4]
) {
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
  const invalidCount = output('invalid-count', 'uint32', 1);
  graph.add(
    new GPUVoronoiDiagram({
      positions: input(
        'positions',
        'float32x2',
        Float32Array.from(positionsValues),
        positionsValues.length / 2
      ),
      triangles: input(
        'triangles',
        'uint32x3',
        Uint32Array.from(triangleValues),
        triangleValues.length / 3
      ),
      triangleCount: input(
        'triangle-count',
        'uint32',
        Uint32Array.of(triangleValues.length / 3),
        1
      ),
      clipBounds: input('clip-bounds', 'float32x4', Float32Array.from(bounds), 1),
      output: {
        segments: segments.view,
        siteIds: siteIds.view,
        status: {
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view,
          invalidCount: invalidCount.view
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
    segments: await readFloat32(segments.buffer, 4 * written),
    siteIds: await readUint32(siteIds.buffer, 2 * written),
    invalidCount: (await readUint32(invalidCount.buffer, 1))[0]
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

it('GPUVoronoiDiagram rejects an unrepresentable final center with invalidCount', async () => {
  // Analytic center is (2^29, approximately 2^140), outside float32 despite finite sites.
  const result = await runVoronoi([-1, -1, 1, 1], [0, 0, 2 ** 30, 0, 2 ** 31, 2 ** -80], [0, 1, 2]);
  expect(result.written).toBe(0);
  expect(result.required).toBe(0);
  expect(result.invalidCount).toBe(1);
});

it('R2 preserves the short-axis bisector for independently scaled edges', async () => {
  const longEdge = 2 ** 40;
  const shortEdge = 2 ** -40;
  const result = await runVoronoi(
    [-longEdge, -shortEdge, longEdge, shortEdge],
    [0, 0, longEdge, 0, 0, shortEdge],
    [0, 1, 2]
  );
  expect(result.invalidCount).toBe(0);
  expect(result.written).toBe(3);
  const row = Array.from({length: result.written}, (_, index) => index).find(
    index => result.siteIds.slice(2 * index, 2 * index + 2).join(',') === '0,2'
  );
  expect(row).toBeDefined();
  for (const component of [1, 3]) {
    const value = result.segments[4 * row! + component];
    expect(Number.isFinite(value)).toBe(true);
    expect(value / shortEdge).toBeCloseTo(0.5, 5);
  }
});

it('R3 keeps a representable center independently of offset and radius overflow', async () => {
  const scale = 2 ** 127;
  const result = await runVoronoi(
    [scale / 2, -scale / 2, 1.5 * scale, scale / 2],
    [-1.5 * scale, 0, -scale, 1.5 * scale, -scale, -1.5 * scale],
    [0, 1, 2]
  );
  expect(result.invalidCount).toBe(0);
  expect(result.written).toBe(3);
  expect(result.required).toBe(3);
  for (let row = 0; row < result.written; row++) {
    expect(Number.isFinite(result.segments[4 * row])).toBe(true);
    expect(Number.isFinite(result.segments[4 * row + 1])).toBe(true);
    expect(result.segments[4 * row] / scale).toBeCloseTo(1, 5);
    expect(result.segments[4 * row + 1] / scale).toBeCloseTo(0, 5);
  }
});

it('R4 emits all hull rays when a finite edge difference overflows, in both windings', async () => {
  const scale = 2 ** 127;
  for (const triangles of [
    [0, 1, 2],
    [0, 2, 1]
  ]) {
    const result = await runVoronoi([-1, -1, 1, 1], [-scale, 0, scale, 0, 0, scale], triangles);
    expect(result.invalidCount).toBe(0);
    expect(result.written).toBe(3);
    expect(result.required).toBe(3);
    const pairs = Array.from({length: result.written}, (_, row) =>
      result.siteIds.slice(2 * row, 2 * row + 2).join(',')
    );
    expect([...pairs].sort()).toEqual(['0,1', '0,2', '1,2']);
    expect(result.segments.slice(4 * pairs.indexOf('0,1'), 4 * pairs.indexOf('0,1') + 4)).toEqual([
      0, 0, 0, -1
    ]);
  }
});

it('R5 clips a visible bisector beyond the former finite ray cap', async () => {
  const result = await runVoronoi([0, -3.25e38, 1, -3.2e38], [0, 0, 1, 0, 0, 1], [0, 1, 2]);
  expect(result.invalidCount).toBe(0);
  expect(result.written).toBe(1);
  expect(result.required).toBe(1);
  expect(result.siteIds).toEqual([0, 1]);
  expect(result.segments[0]).toBe(0.5);
  expect(result.segments[2]).toBe(0.5);
  expect(Number.isFinite(result.segments[1])).toBe(true);
  expect(Number.isFinite(result.segments[3])).toBe(true);
  expect(result.segments[1] / Math.fround(-3.2e38)).toBeCloseTo(1, 5);
  expect(result.segments[3] / Math.fround(-3.25e38)).toBeCloseTo(1, 5);
});

for (const exponent of [0, 45, -52, 100, -100]) {
  it(`GPUVoronoiDiagram emits the analytic clipped bisectors for either winding at scale 2^${exponent}`, async () => {
    const scale = 2 ** exponent;
    const expectedByPair: Record<string, number[]> = {
      '0,1': [0.5, 0.5, 0.5, -1],
      '1,2': [0.5, 0.5, 2, 2],
      '0,2': [0.5, 0.5, -1, 0.5]
    };
    for (const triangle of [
      [0, 1, 2],
      [0, 2, 1]
    ]) {
      const result = await runVoronoi(
        [-scale, -scale, 2 * scale, 2 * scale],
        [0, 0, scale, 0, 0, scale],
        triangle
      );
      expect(result.written).toBe(3);
      expect(result.required).toBe(3);
      expect(result.invalidCount).toBe(0);
      const seen = new Set<string>();
      for (let row = 0; row < result.written; row++) {
        const pair = result.siteIds.slice(2 * row, 2 * row + 2).join(',');
        seen.add(pair);
        expect(expectedByPair[pair]).toBeDefined();
        result.segments.slice(4 * row, 4 * row + 4).forEach((value, component) => {
          expect(Number.isFinite(value)).toBe(true);
          expect(value / scale).toBeCloseTo(expectedByPair[pair][component], 5);
        });
      }
      expect(Array.from(seen).sort()).toEqual(['0,1', '0,2', '1,2']);
    }
  });
}
