// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// DGGS-105: ring orientation. Before the fix the query pass read the winding sum with its sign, so
// a clockwise hole (GeoJSON order) was ignored and a clockwise shell contained nothing, both with
// status 0. Expectations are hand-worked: every ring is an axis-aligned square in the gnomonic
// chart x = 1 (chart coordinates (y, z)), the shell has half-width 0.3 and the hole 0.1, so a chart
// point is inside the shell when max(|y|, |z|) < 0.3 and inside the hole when it is below 0.1.

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  OFF_AXIS_CLOCKWISE_TRIANGLE,
  OFF_AXIS_INTERIOR_QUERY,
  OFF_AXIS_SHIFTED_INTERIOR_QUERY
} from './off-axis-ring-fixture';
import {createGlobalCoordinateFixture} from './global-coordinate-fixture';

it('off-axis tiny rings contain their interior and exclude the antipode in both orientations', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reversed of [false, true])
    for (const complement of [false, true]) {
      const fixture = createGlobalCoordinateFixture(device);
      const vertices = reversed
        ? [...OFF_AXIS_CLOCKWISE_TRIANGLE].reverse()
        : OFF_AXIS_CLOCKWISE_TRIANGLE;
      const contains = fixture.output('uint32', 4);
      const status = fixture.output('uint32', 4);
      fixture.graph.add(
        new GPUSphericalPolygonContains({
          positions: fixture.input('float32x3', vertices.flat()),
          ringOffsets: fixture.input('uint32', [0, 3]),
          polygonOffsets: fixture.input('uint32', [0, 1]),
          queries: fixture.input(
            'float32x3',
            [OFF_AXIS_INTERIOR_QUERY, OFF_AXIS_SHIFTED_INTERIOR_QUERY].flatMap(point => [
              ...point,
              ...point.map(value => -value)
            ])
          ),
          polygonIndices: fixture.input('uint32', [0, 0, 0, 0]),
          complement,
          output: {contains: contains.view, status: status.view}
        })
      );
      const compiled = fixture.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(status.buffer, 4)).toEqual([0, 0, 0, 0]);
        expect(await readUint32(contains.buffer, 4)).toEqual(
          complement ? [0, 1, 0, 1] : [1, 0, 1, 0]
        );
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    }
});

function normalizePoints(points: number[][]): number[] {
  return points.flatMap(point => {
    const length = Math.hypot(...point);
    return point.map(value => value / length);
  });
}

/** Square of half-width `size` in the chart x = 1, counterclockwise seen from outside (+x). */
function getCounterclockwiseSquare(size: number): number[][] {
  return [
    [1, -size, -size],
    [1, size, -size],
    [1, size, size],
    [1, -size, size]
  ];
}

function getClockwiseSquare(size: number): number[][] {
  return getCounterclockwiseSquare(size).reverse();
}

it('GPUSphericalPolygonContains gives the same region for clockwise and counterclockwise rings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Polygon 0: CCW shell, CCW hole. 1: CCW shell, CW hole (GeoJSON). 2: CW shell alone.
  // 3: CW shell, CCW hole.
  const rings = [
    getCounterclockwiseSquare(0.3),
    getCounterclockwiseSquare(0.1),
    getCounterclockwiseSquare(0.3),
    getClockwiseSquare(0.1),
    getClockwiseSquare(0.3),
    getClockwiseSquare(0.3),
    getCounterclockwiseSquare(0.1)
  ];
  const ringOffsets = [0];
  for (const ring of rings) ringOffsets.push(ringOffsets[ringOffsets.length - 1] + ring.length);
  const polygonOffsets = [0, 2, 4, 5, 7];
  const hasHole = [true, true, false, true];
  // Per polygon: inside the hole square, in the shell band, outside the shell, and the antipodal
  // image of the band point (the winding there is -2 pi for a counterclockwise ring).
  const queryPoints = [
    [1, 0.02, -0.03],
    [1, 0.2, 0.05],
    [1, 0.5, 0],
    [-1, -0.2, -0.05]
  ];
  const expectedPerPolygon = (polygon: number) => [hasHole[polygon] ? 0 : 1, 1, 0, 0];
  const polygonCount = polygonOffsets.length - 1;
  const queryCount = polygonCount * queryPoints.length;
  const queries = normalizePoints(Array.from({length: polygonCount}, () => queryPoints).flat());
  const polygonIndices = Array.from({length: queryCount}, (_, query) =>
    Math.floor(query / queryPoints.length)
  );
  const vertexCount = ringOffsets[ringOffsets.length - 1];
  const graph = new GPUCommandGraph(device);
  const buffers = [
    createInputBuffer(device, new Float32Array(normalizePoints(rings.flat()))),
    createInputBuffer(device, new Uint32Array(ringOffsets)),
    createInputBuffer(device, new Uint32Array(polygonOffsets)),
    createInputBuffer(device, new Float32Array(queries)),
    createInputBuffer(device, new Uint32Array(polygonIndices)),
    createOutputBuffer(device, queryCount),
    createOutputBuffer(device, queryCount)
  ];
  const view = <Format extends 'float32x3' | 'uint32'>(
    index: number,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, `buffer-${index}`, buffers[index], format, length);
  graph.add(
    new GPUSphericalPolygonContains({
      positions: view(0, 'float32x3', vertexCount),
      ringOffsets: view(1, 'uint32', ringOffsets.length),
      polygonOffsets: view(2, 'uint32', polygonOffsets.length),
      queries: view(3, 'float32x3', queryCount),
      polygonIndices: view(4, 'uint32', queryCount),
      output: {contains: view(5, 'uint32', queryCount), status: view(6, 'uint32', queryCount)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const status = await readUint32(buffers[6], queryCount);
    const contains = await readUint32(buffers[5], queryCount);
    expect(status).toEqual(new Array(queryCount).fill(0));
    for (let polygon = 0; polygon < polygonCount; polygon++) {
      const begin = polygon * queryPoints.length;
      expect(contains.slice(begin, begin + queryPoints.length), `polygon ${polygon}`).toEqual(
        expectedPerPolygon(polygon)
      );
    }
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
