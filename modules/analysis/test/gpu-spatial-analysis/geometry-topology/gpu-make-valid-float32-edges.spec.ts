// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {runMakeValid, type MakeValidResult, type Point} from './topology-property-harness';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review 2026-11-10, polygon-topology-01 and -04. Expected values are hand
// computed: each input is a valid simple ring, so make-valid returns that one polygon with the
// ring's own area (a rectangle with one extra vertex on a side, or a plain square).

const CAPACITY = {segments: 64, intersections: 64, rings: 16, vertices: 128};

/** Polygon area (shell minus holes) from a float64 shoelace sum about `origin`. */
function getPolygonArea(rings: readonly number[][][], origin: Point): number {
  let area = 0;
  rings.forEach((ring, ringIndex) => {
    let doubled = 0;
    for (let index = 0; index < ring.length; index++) {
      const [x0, y0] = ring[index];
      const [x1, y1] = ring[(index + 1) % ring.length];
      doubled += (x0 - origin[0]) * (y1 - origin[1]) - (x1 - origin[0]) * (y0 - origin[1]);
    }
    area += (ringIndex === 0 ? 1 : -1) * Math.abs(doubled / 2);
  });
  return area;
}

function expectOnePolygon(result: MakeValidResult, area: number, origin: Point): void {
  expect(result.overflow).toBe(0);
  expect(result.count).toBe(1);
  expect(result.polygons[0].length).toBe(1);
  expect(getPolygonArea(result.polygons[0], origin)).toBeCloseTo(area, 4);
}

it('GPUMakeValid keeps a face whose first shell edge is one float32 step tall (review 01)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The report's ring: the first edge runs from (0, 1 + 2^-23) down to (0, 1). Its float32
  // midpoint rounds to the lower endpoint. The vertex tolerance is below one float32 step so the
  // two vertices stay distinct.
  const step = 2 ** -23;
  const unitRing: Point[] = [
    [0, 1 + step],
    [0, 1],
    [5, 1],
    [5, 7],
    [0, 7],
    [0, 1 + step]
  ];
  expectOnePolygon(
    await runMakeValid(device, [unitRing], {...CAPACITY, vertexTolerance: 2 ** -26}),
    30,
    [0, 1]
  );
  // The same shape at a projected northing, where one float32 step is 0.25 units: an ordinary
  // near-duplicate vertex of float64 data.
  const northing = 4000000;
  const projectedRing: Point[] = [
    [0, northing + 0.25],
    [0, northing],
    [5, northing],
    [5, northing + 7],
    [0, northing + 7],
    [0, northing + 0.25]
  ];
  expectOnePolygon(await runMakeValid(device, [projectedRing], CAPACITY), 35, [0, northing]);
  // Started one vertex later the first edge is the long bottom side; this already passed.
  const rotatedRing: Point[] = [...projectedRing.slice(1, 5), projectedRing[1]];
  expectOnePolygon(await runMakeValid(device, [rotatedRing], CAPACITY), 35, [0, northing]);
  device.destroy?.();
});

it('GPUMakeValid closes a ring given without its repeated first vertex (review 04)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const openSquare: Point[] = [
    [0, 0],
    [4, 0],
    [4, 4],
    [0, 4]
  ];
  expectOnePolygon(await runMakeValid(device, [openSquare], CAPACITY), 16, [0, 0]);
  // A square with a square hole, both rings implicitly closed: the annulus, area 16 - 4.
  const openHole: Point[] = [
    [1, 1],
    [1, 3],
    [3, 3],
    [3, 1]
  ];
  const annulus = await runMakeValid(device, [openSquare, openHole], CAPACITY);
  expect(annulus.overflow).toBe(0);
  expect(annulus.count).toBe(1);
  expect(annulus.polygons[0].length).toBe(2);
  expect(getPolygonArea(annulus.polygons[0], [0, 0])).toBeCloseTo(12, 4);
  // The explicitly closed form gives the same polygon.
  expectOnePolygon(
    await runMakeValid(device, [[...openSquare, openSquare[0]]], CAPACITY),
    16,
    [0, 0]
  );
  device.destroy?.();
});

it('GPUMakeValid reads a repeated vertex as one vertex', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A zero-length noded row used to be a self-loop that stranded the rings through its vertex:
  // the lens scene of gpu-geometry-topology.spec.ts lost a polygon once each ring ended in a
  // second copy of its first vertex. Hand values: the square (area 16) minus the lens, 14, and
  // the lobe of the second ring outside the square, 2.
  const shell: Point[] = [
    [0, 0],
    [4, 0],
    [4, 0],
    [4, 4],
    [0, 4],
    [0, 0],
    [0, 0]
  ];
  const crossingRing: Point[] = [
    [3, 1],
    [5, 1],
    [5, 3],
    [5, 3],
    [3, 3],
    [3, 1],
    [3, 1]
  ];
  expectOnePolygon(await runMakeValid(device, [shell], CAPACITY), 16, [0, 0]);
  const result = await runMakeValid(device, [shell, crossingRing], CAPACITY);
  expect(result.overflow).toBe(0);
  expect(result.count).toBe(2);
  const areas = result.polygons.map(rings => getPolygonArea(rings, [0, 0])).sort((a, b) => a - b);
  expect(areas[0]).toBeCloseTo(2, 4);
  expect(areas[1]).toBeCloseTo(14, 4);
  device.destroy?.();
});
