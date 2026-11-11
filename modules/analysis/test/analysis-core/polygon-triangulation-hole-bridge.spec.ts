// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import earcut from 'earcut';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUPolygonTriangulation,
  getPolygonTriangulationIndexCount
} from '../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

// Hole bridges of the ear-clipping path when two shell vertices lie on one ray from the hole's
// leftmost vertex (equal tangent), for both orders in which the walk can meet them. The bridge
// takes the nearer vertex, as Earcut does (closing pass of the math correctness review,
// 2026-11-11: the tie used to go to any later candidate right of the first one, which on other
// polygons ran the bridge through the nearer vertex and left a ring the ear clipping could not
// finish; `gpu-polygon-triangulation-search.spec.ts` pins those). The expected values do not come
// from the kernel: areas sum to the shoelace area of shell minus hole, every triangle lies
// inside the shell and outside the hole, and the triangle count is that of the `earcut` package
// (3.2.4, a test oracle only) on the same coordinates.

type Point = [number, number];

function getSignedArea(ring: Point[]): number {
  let twiceArea = 0;
  for (let index = 0; index < ring.length; index++) {
    const [x0, y0] = ring[index];
    const [x1, y1] = ring[(index + 1) % ring.length];
    twiceArea += x0 * y1 - x1 * y0;
  }
  return twiceArea / 2;
}

/** Even-odd containment of a point that is not on the ring. */
function isInsideRing(point: Point, ring: Point[]): boolean {
  let inside = false;
  for (let index = 0; index < ring.length; index++) {
    const [x0, y0] = ring[index];
    const [x1, y1] = ring[(index + 1) % ring.length];
    if (y0 > point[1] !== y1 > point[1]) {
      const crossing = x0 + ((point[1] - y0) * (x1 - x0)) / (y1 - y0);
      if (crossing > point[0]) {
        inside = !inside;
      }
    }
  }
  return inside;
}

async function triangulate(
  shell: Point[],
  hole: Point[]
): Promise<{triangles: number[][]; valid: number}> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'hole-bridge-triangulation'});
  const vertexCount = shell.length + hole.length;
  const indexCount = getPolygonTriangulationIndexCount(vertexCount, 2, 1);
  const buffers = {
    positions: createInputBuffer(device, new Float32Array([...shell.flat(), ...hole.flat()])),
    featureOffsets: createInputBuffer(device, new Uint32Array([0, 1])),
    polygonOffsets: createInputBuffer(device, new Uint32Array([0, 2])),
    ringOffsets: createInputBuffer(device, new Uint32Array([0, shell.length, vertexCount])),
    indices: createOutputBuffer(device, indexCount),
    valid: createOutputBuffer(device, 1),
    triangleCount: createOutputBuffer(device, 1)
  };
  try {
    graph.add(
      new GPUPolygonTriangulation({
        positions: importGraphBuffer(
          graph,
          'positions',
          buffers.positions,
          'float32x2',
          vertexCount
        ),
        featureOffsets: importGraphBuffer(graph, 'features', buffers.featureOffsets, 'uint32', 2),
        polygonOffsets: importGraphBuffer(graph, 'polygons', buffers.polygonOffsets, 'uint32', 2),
        ringOffsets: importGraphBuffer(graph, 'rings', buffers.ringOffsets, 'uint32', 3),
        largePolygonMethod: 'none',
        output: {
          indices: importGraphBuffer(graph, 'indices', buffers.indices, 'uint32', indexCount),
          valid: importGraphBuffer(graph, 'valid', buffers.valid, 'uint32', 1),
          triangleCount: importGraphBuffer(graph, 'counts', buffers.triangleCount, 'uint32', 1)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(buffers.triangleCount, 1);
      const indices = await readUint32(buffers.indices, indexCount);
      const [valid] = await readUint32(buffers.valid, 1);
      const triangles: number[][] = [];
      for (let triangle = 0; triangle < count; triangle++) {
        triangles.push(Array.from(indices.slice(3 * triangle, 3 * triangle + 3)));
      }
      return {triangles, valid};
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of Object.values(buffers)) {
      buffer.destroy();
    }
  }
}

function expectTiling(shell: Point[], hole: Point[], triangles: number[][]): void {
  const points = [...shell, ...hole];
  const expectedArea = Math.abs(getSignedArea(shell)) - Math.abs(getSignedArea(hole));
  let area = 0;
  for (const triangle of triangles) {
    const corners = triangle.map(index => points[index]);
    const triangleArea = Math.abs(getSignedArea(corners));
    expect(triangleArea, `triangle ${triangle}`).toBeGreaterThan(0);
    area += triangleArea;
    const centroid: Point = [
      (corners[0][0] + corners[1][0] + corners[2][0]) / 3,
      (corners[0][1] + corners[1][1] + corners[2][1]) / 3
    ];
    expect(isInsideRing(centroid, shell), `triangle ${triangle} inside the shell`).toBe(true);
    expect(isInsideRing(centroid, hole), `triangle ${triangle} outside the hole`).toBe(false);
  }
  // Coordinates are small integers, so the areas are exact halves.
  expect(area).toBe(expectedArea);
  const referenceIndices = earcut([...shell.flat(), ...hole.flat()], [shell.length]);
  expect(triangles.length).toBe(referenceIndices.length / 3);
}

it('a hole whose bridge has two candidates on one ray still tiles the polygon, in either visiting order', async () => {
  const hole: Point[] = [
    [10, 5],
    [12, 4],
    [12, 6]
  ];
  // Two notches cut down from the top edge. Their tips (6, 6) and (2, 7) are collinear with the
  // hole's leftmost vertex (10, 5): |dy| / dx = 1 / 4 = 2 / 8. The ray to the left from (10, 5)
  // meets the slanted left edge, whose upper end (-1, 10) is the first bridge candidate, and the
  // walk from there meets the nearer tip (6, 6) before the farther tip (2, 7).
  const shellNearFirst: Point[] = [
    [-1, 10],
    [0, 0],
    [20, 0],
    [20, 10],
    [7, 10],
    [6, 6],
    [5, 10],
    [3, 10],
    [2, 7],
    [1, 10]
  ];
  const nearFirst = await triangulate(shellNearFirst, hole);
  expect(nearFirst.valid).toBe(1);
  expectTiling(shellNearFirst, hole, nearFirst.triangles);
  // Mirror in y: notches rise from the bottom edge and the walk meets the farther tip first.
  const mirror = (ring: Point[]): Point[] => ring.map(([x, y]): Point => [x, 10 - y]);
  const shellFarFirst = mirror(shellNearFirst);
  const mirroredHole = mirror(hole);
  const farFirst = await triangulate(shellFarFirst, mirroredHole);
  expect(farFirst.valid).toBe(1);
  expectTiling(shellFarFirst, mirroredHole, farFirst.triangles);
});
