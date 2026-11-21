// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/index';
import {expectGeographicGroupBounds} from '../geometry-refinement/geometry-output-review-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

async function measureLines(positions: number[], offsets: number[]) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const components = format === 'float32x4' ? 4 : format === 'float32x2' ? 2 : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const featureCount = offsets.length - 1;
  const bounds = output('bounds', 'float32x4', featureCount);
  const centroids = output('centroids', 'float32x2', featureCount);
  const groupBounds = output('group-bounds', 'float32x4', 1);
  const groupCentroids = output('group-centroids', 'float32x2', 1);
  const counts = output('feature-counts', 'uint32', 1),
    vertices = output('vertex-counts', 'uint32', 1);
  graph.add(
    new GPUGeometryMeasures({
      geometryType: 'lines',
      positions: input('positions', 'float32x2', Float32Array.from(positions)),
      lineOffsets: input('line-offsets', 'uint32', Uint32Array.from(offsets)),
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      },
      groupIds: input('group-ids', 'uint32', new Uint32Array(featureCount)),
      groupCount: 1,
      output: {bounds: bounds.view, centroids: centroids.view},
      groupOutput: {
        bounds: groupBounds.view,
        centroids: groupCentroids.view,
        featureCounts: counts.view,
        vertexCounts: vertices.view
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      bounds: await readFloat32(bounds.buffer, featureCount * 4),
      centroids: await readFloat32(centroids.buffer, featureCount * 2),
      groupBounds: await readFloat32(groupBounds.buffer, 4),
      groupCentroids: await readFloat32(groupCentroids.buffer, 2),
      counts: await readUint32(counts.buffer, 1),
      vertices: await readUint32(vertices.buffer, 1)
    };
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
}

it.each([
  -179.5, -178.5
])('geographic group bounds include singleton longitude %s on the first feature branch', async singletonLongitude => {
  // The first singleton stays within [179,181]; the companion must extend maxX to 181.5.
  // This distinguishes ignoring singleton bounds from failing to align their longitude branch.
  const result = await measureLines([179, 0, -179, 1, singletonLongitude, 0.5], [0, 2, 3]);
  expectGeographicGroupBounds(result, singletonLongitude);
});

it('geographic group bounds retain a duplicate endpoint already contained by the first interval', async () => {
  const result = await measureLines([0, 0, -100, 1, 0, 2, 100, 3, 100, 3], [0, 4, 5]);
  // First interval [-100,100] already contains its terminal longitude 100; no turn is needed.
  expect(result.bounds).toEqual([-100, 0, 100, 3, 100, 3, 100, 3]);
  expect(result.groupBounds).toEqual([-100, 0, 100, 3]);
  expect(result.counts).toEqual([2]);
  expect(result.vertices).toEqual([5]);
});

it('geographic feature centroids stay unwrapped while group centroids use the principal branch', async () => {
  // Nondegenerate equatorial arc: the midpoint of 540 -> 542 is 541, equivalent to -179.
  const result = await measureLines([540, 0, 542, 0], [0, 2]);
  for (const value of [...result.centroids, ...result.groupCentroids])
    expect(Number.isFinite(value)).toBe(true);
  expect(result.centroids[0]).toBeCloseTo(541, 3);
  expect(result.centroids[1]).toBeCloseTo(0, 5);
  expect(result.groupCentroids[0]).toBeCloseTo(-179, 3);
  expect(result.groupCentroids[1]).toBeCloseTo(0, 5);
  expect(result.counts).toEqual([1]);
  expect(result.vertices).toEqual([2]);
});
