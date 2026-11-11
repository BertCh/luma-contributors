// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Boundary rule of GPURegionMask (math review cells-joins-03): a point on the boundary of the
// region is selected, for a rectangle and for a lasso polygon alike, as published
// `GPUPointSpatialQuery` documents for bounds and polygons. Expected masks are written by hand.

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPURegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Region =
  | {kind: 'rectangle'; bounds: number[]}
  | {kind: 'polygon'; vertices: number[]; vertexCount?: number};

async function runMask(points: [number, number][], region: Region): Promise<number[]> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = points.length;
  const positionsBuffer = createInputBuffer(device, new Float32Array(points.flat()));
  const maskBuffer = createOutputBuffer(device, rowCount);
  const overflowBuffer = createOutputBuffer(device, 1);
  const shapeBuffer = createInputBuffer(
    device,
    new Float32Array(region.kind === 'rectangle' ? region.bounds : region.vertices)
  );
  const countBuffer = createInputBuffer(
    device,
    Uint32Array.of(
      region.kind === 'polygon' ? (region.vertexCount ?? region.vertices.length / 2) : 0
    )
  );
  const graph = new GPUCommandGraph(device, {id: 'region-mask-boundary'});
  graph.add(
    new GPURegionMask({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
      region:
        region.kind === 'rectangle'
          ? {
              kind: 'rectangle',
              bounds: importGraphBuffer(graph, 'bounds', shapeBuffer, 'float32', 4)
            }
          : {
              kind: 'polygon',
              vertices: importGraphBuffer(
                graph,
                'vertices',
                shapeBuffer,
                'float32x2',
                region.vertices.length / 2
              ),
              vertexCount: importGraphBuffer(graph, 'vertex-count', countBuffer, 'uint32', 1)
            },
      output: {
        mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rowCount),
        status: {overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const mask = await readUint32(maskBuffer, rowCount);
  compiled.destroy();
  for (const buffer of [positionsBuffer, maskBuffer, overflowBuffer, shapeBuffer, countBuffer]) {
    buffer.destroy();
  }
  return mask;
}

it('GPURegionMask selects boundary points of a rectangle and of the same square as a lasso', async () => {
  const points: [number, number][] = [
    [1, 0.5], // right edge
    [0, 0.5], // left edge
    [0.5, 0], // bottom edge
    [0.5, 1], // top edge
    [0, 0], // corners
    [1, 0],
    [1, 1],
    [0, 1],
    [0.5, 0.5], // inside
    [1.5, 0.5], // outside, level with the square
    [-0.5, 0.5],
    [0.5, 1.5],
    [0.5, -0.5],
    [2, 1], // outside, on the line of the top edge
    [1, 2], // outside, on the line of the right edge
    [2, 2] // outside, on the diagonal
  ];
  const expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
  expect(await runMask(points, {kind: 'rectangle', bounds: [0, 0, 1, 1]})).toEqual(expected);
  const counterClockwise = [0, 0, 1, 0, 1, 1, 0, 1];
  const clockwise = [0, 0, 0, 1, 1, 1, 1, 0];
  expect(await runMask(points, {kind: 'polygon', vertices: counterClockwise})).toEqual(expected);
  expect(await runMask(points, {kind: 'polygon', vertices: clockwise})).toEqual(expected);
});

it('GPURegionMask selects points on oblique and horizontal lasso edges, exactly', async () => {
  // Triangle (0, 0), (4, 2), (0, 4), then an L-shape whose reflex corner and collinear edges
  // exercise horizontal edges at the level of the point.
  const triangle = [0, 0, 4, 2, 0, 4];
  const trianglePoints: [number, number][] = [
    [2, 1], // on the edge (0, 0)-(4, 2)
    [2, 3], // on the edge (4, 2)-(0, 4)
    [0, 2], // on the vertical edge
    [4, 2], // the apex vertex
    [1, 2], // inside
    [2, 0.99999994], // one float32 step below the oblique edge: outside
    [2, 1.0000001], // one float32 step above it: inside
    [6, 3], // on the line of the first edge, beyond its end
    [-2, -1], // on the same line, before its start
    [4.0000005, 2], // one float32 step right of the apex
    [5, 2] // level with the apex, outside
  ];
  expect(await runMask(trianglePoints, {kind: 'polygon', vertices: triangle})).toEqual([
    1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0
  ]);

  const letterL = [0, 0, 4, 0, 4, 2, 2, 2, 2, 4, 0, 4];
  const letterPoints: [number, number][] = [
    [3, 2], // on the horizontal edge (4, 2)-(2, 2)
    [2, 2], // the reflex corner
    [2, 3], // on the vertical edge (2, 2)-(2, 4)
    [1, 2], // inside, level with the horizontal edge
    [3, 3], // in the notch: outside
    [5, 2], // outside, on the line of the horizontal edge
    [1, 4], // on the top edge
    [3, 4], // outside, on the line of the top edge
    [-1, 2], // outside, level with the horizontal edge
    [-1, 0] // outside, on the line of the bottom edge
  ];
  expect(await runMask(letterPoints, {kind: 'polygon', vertices: letterL})).toEqual([
    1, 1, 1, 1, 0, 0, 1, 0, 0, 0
  ]);
  // Only the first three vertices are active: the triangle (0, 0), (4, 0), (4, 2).
  expect(
    await runMask(
      [
        [2, 1], // on its closing edge (4, 2)-(0, 0)
        [3, 1], // inside
        [1, 1], // outside
        [3, 2] // on an inactive edge
      ],
      {kind: 'polygon', vertices: letterL, vertexCount: 3}
    )
  ).toEqual([1, 1, 0, 0]);
});
