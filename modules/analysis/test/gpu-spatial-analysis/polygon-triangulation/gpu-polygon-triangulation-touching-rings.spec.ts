// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// `GPUPolygonTriangulation` on valid polygons whose rings touch at single points: vertex to
// vertex, or a vertex of one ring inside an edge of another. Review fixes of the closing pass of
// the math correctness review (2026-11-11), findings frc-geometry-01 to 04 of
// `dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-geometry.md`.
//
// What is expected does not come from the kernel:
// - Validity and area are Shapely's (`Polygon.is_valid`, `Polygon.area`, Shapely 2.2.0 on GEOS
//   3.14.1), written by `test/oracles/polygon-triangulation-touching-rings.py`.
// - `getTilingDefect` decides exactly whether the triangles tile the polygon.
//
// The rule under test: a result flagged `valid = 1` is a tiling. Before these fixes the Earcut
// port returned `valid = 1` with triangles that overlapped a hole for some of these polygons, the
// monotone path did the same for others, and both flagged polygons that they can tile. The
// entries marked `failedBefore` and the review inputs are the ones that did.

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getPolygonTriangulationNodeCount} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  METRE_RESOLUTION_FRAME,
  getBoundaryChainDefect,
  getTilingDefect,
  getTriangleAreaSum,
  mapToFrame,
  type SearchPoint,
  type SearchPolygon
} from './polygon-triangulation-search-cases';
import {EARCUT_ONLY, MONOTONE_ONLY, triangulateBatch} from './polygon-triangulation-search-harness';
import fixtureJSON from '../../oracles/fixtures/polygon-triangulation-touching-rings.json';

type FixturePolygon = {
  name: string;
  /** Shell first, then holes; integer coordinates; any winding. */
  rings: SearchPoint[][];
  /** `shapely.Polygon(shell, holes).area`. */
  area: number;
  /** Whether two rings of the polygon intersect (Shapely). */
  touching: boolean;
  /** Flagged or triangulated wrongly by the kernel as it was before the review fixes. */
  failedBefore?: boolean;
};

const FIXTURE = fixtureJSON as unknown as {
  generatedCount: number;
  review: FixturePolygon[];
  families: FixturePolygon[];
};

const ALL_POLYGONS: FixturePolygon[] = [...FIXTURE.review, ...FIXTURE.families];

it('the scratch reserves one node per hole for a vertex inserted into a touched edge', () => {
  // 100 vertices in 5 rings of 2 polygons: 3 holes. Twice the nodes of the rings and their
  // bridges, 8 per polygon, and one per hole.
  expect(getPolygonTriangulationNodeCount(100, 5, 2)).toBe(2 * (100 + 2 * 3) + 8 * 2 + 3);
  // Without holes the count is what it was.
  expect(getPolygonTriangulationNodeCount(100, 2, 2)).toBe(2 * 100 + 8 * 2);
});

it('the fixture holds valid polygons with touching rings, among them those that failed before', () => {
  expect(FIXTURE.generatedCount).toBe(7938);
  expect(FIXTURE.review.length).toBe(16);
  expect(FIXTURE.families.length).toBe(1043);
  expect(FIXTURE.families.filter(polygon => polygon.failedBefore).length).toBe(59);
  expect(ALL_POLYGONS.filter(polygon => polygon.touching).length).toBe(950);
});

it('the Earcut port tiles every valid polygon whose rings touch (Shapely validity and area)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const options of [EARCUT_ONLY, {}]) {
    const label = options === EARCUT_ONLY ? 'none' : 'default';
    const result = await triangulateBatch(device, ALL_POLYGONS, options);
    const departures: string[] = [];
    ALL_POLYGONS.forEach((polygon, index) => {
      if (result.valid[index] !== 1) {
        departures.push(`${polygon.name}: valid = 0`);
        return;
      }
      const defect = getTilingDefect(polygon.rings, result.triangles[index]);
      if (defect) {
        departures.push(`${polygon.name}: valid = 1 and ${defect}`);
      } else if (getTriangleAreaSum(polygon.rings, result.triangles[index]) !== polygon.area) {
        departures.push(`${polygon.name}: the triangle areas do not sum to ${polygon.area}`);
      }
    });
    expect(departures, label).toEqual([]);
  }
}, 300_000);

it('the Earcut port tiles the same polygons at float32 resolution of projected metres', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The sheared families do not fit this frame (one lattice step is one float32 step here).
  const polygons: SearchPolygon[] = ALL_POLYGONS.filter(polygon =>
    polygon.rings.flat(2).every(value => Math.abs(value) <= 64)
  ).map(polygon => ({
    name: polygon.name,
    rings: mapToFrame(METRE_RESOLUTION_FRAME, polygon.rings)
  }));
  expect(polygons.length).toBeGreaterThan(500);
  const result = await triangulateBatch(device, polygons, EARCUT_ONLY);
  const departures: string[] = [];
  polygons.forEach((polygon, index) => {
    const defect =
      result.valid[index] === 1
        ? getTilingDefect(polygon.rings, result.triangles[index])
        : 'valid = 0';
    if (defect) {
      departures.push(`${polygon.name}: ${defect}`);
    }
  });
  expect(departures).toEqual([]);
}, 300_000);

it('the monotone path alone flags a polygon or tiles it, and never returns a wrong triangulation as valid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await triangulateBatch(device, ALL_POLYGONS, MONOTONE_ONLY);
  const departures: string[] = [];
  let flaggedCount = 0;
  let validTouchingCount = 0;
  ALL_POLYGONS.forEach((polygon, index) => {
    if (result.valid[index] !== 1) {
      flaggedCount++;
      // Rings that do not touch are within what the path supports.
      if (!polygon.touching) {
        departures.push(`${polygon.name}: valid = 0 and the rings do not touch`);
      }
      return;
    }
    validTouchingCount += polygon.touching ? 1 : 0;
    // The path keeps collinear vertices, so it may write triangles of zero area.
    const defect = getTilingDefect(polygon.rings, result.triangles[index], true);
    if (defect) {
      departures.push(`${polygon.name}: valid = 1 and ${defect}`);
    }
  });
  expect(departures).toEqual([]);
  // Most polygons with touching rings are beyond the path, some are not.
  expect(flaggedCount).toBeGreaterThan(500);
  expect(validTouchingCount).toBeGreaterThan(50);
}, 300_000);

it('a hole outside its shell or across it, or a coordinate that is not finite, flags the polygon', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const square: SearchPoint[] = [
    [0, 0],
    [4, 0],
    [4, 4],
    [0, 4]
  ];
  const polygons: SearchPolygon[] = [
    // No shell edge is left of the hole, so there is no bridge. Before: `valid = 1` with the two
    // triangles of the square and the hole dropped, on both methods.
    {
      name: 'hole outside the shell',
      rings: [
        square,
        [
          [-5, 1],
          [-4, 2],
          [-4, 1]
        ]
      ]
    },
    // Before: `valid = 0` from Earcut, then `valid = 1` with seven triangles from the monotone
    // retry of the default method, one of them clockwise.
    {
      name: 'hole across a shell edge',
      rings: [
        square,
        [
          [3, 1],
          [6, 2],
          [3, 3]
        ]
      ]
    },
    // No orientation with this vertex can be certified. Before: `valid = 1` with one triangle
    // from Earcut (the monotone path alone flagged it).
    {
      name: 'shell vertex that is not finite',
      rings: [
        [
          [0, 0],
          [4, 0],
          [Number.POSITIVE_INFINITY, 4],
          [0, 4]
        ]
      ]
    }
  ];
  for (const options of [EARCUT_ONLY, {}, MONOTONE_ONLY]) {
    const result = await triangulateBatch(device, polygons, options);
    expect(result.valid).toEqual([0, 0, 0]);
    expect(result.triangles).toEqual([[], [], []]);
  }
}, 120_000);

/**
 * The polygon scaled by 1024, with every shell edge divided into equal parts by collinear
 * vertices so that the polygon has at least 4096 vertices and skips Earcut in the default
 * configuration. The parts are a power of two, so every new vertex is on the lattice.
 */
function enlargePolygon(polygon: FixturePolygon): SearchPolygon {
  const [shell, ...holes] = polygon.rings.map(ring =>
    ring.map(([x, y]): SearchPoint => [1024 * x, 1024 * y])
  );
  let partCount = 1;
  while (partCount * shell.length < 4096) {
    partCount *= 2;
  }
  const divided: SearchPoint[] = [];
  shell.forEach((start, index) => {
    const end = shell[(index + 1) % shell.length];
    for (let part = 0; part < partCount; part++) {
      divided.push([
        start[0] + ((end[0] - start[0]) / partCount) * part,
        start[1] + ((end[1] - start[1]) / partCount) * part
      ]);
    }
  });
  return {name: `${polygon.name}, enlarged`, rings: [divided, ...holes]};
}

it('the default method tiles polygons of more than 4096 vertices whose rings touch, as method none does', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = FIXTURE.review.map(enlargePolygon);
  for (const polygon of polygons) {
    const vertexCount = polygon.rings.reduce((sum, ring) => sum + ring.length, 0);
    expect(vertexCount, polygon.name).toBeGreaterThanOrEqual(4096);
    for (const value of polygon.rings.flat(2)) {
      expect(Number.isInteger(value), polygon.name).toBe(true);
    }
  }
  for (const options of [{}, EARCUT_ONLY]) {
    const label = options === EARCUT_ONLY ? 'none' : 'default';
    const result = await triangulateBatch(device, polygons, options);
    const departures: string[] = [];
    polygons.forEach((polygon, index) => {
      // Polygons of this size take the monotone path first, which keeps the collinear vertices
      // of the shell and writes triangles of zero area between them.
      const defect =
        result.valid[index] === 1
          ? getBoundaryChainDefect(polygon.rings, result.triangles[index], true)
          : 'valid = 0';
      const expectedArea = 1024 * 1024 * FIXTURE.review[index].area;
      if (defect) {
        departures.push(`${polygon.name}: ${defect}`);
      } else if (getTriangleAreaSum(polygon.rings, result.triangles[index]) !== expectedArea) {
        departures.push(`${polygon.name}: the triangle areas do not sum to ${expectedArea}`);
      }
    });
    expect(departures, label).toEqual([]);
  }
}, 300_000);
