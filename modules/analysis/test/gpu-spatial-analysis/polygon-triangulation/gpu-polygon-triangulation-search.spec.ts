// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Randomized search of `GPUPolygonTriangulation` on valid polygons with holes, in batches on the
// GPU, for both paths: the Earcut port (`largePolygonMethod: 'none'`) and the monotone path alone.
//
// What is expected does not come from the kernel:
// - `getTilingDefect` (exact arithmetic on the float32 coordinates) says whether the triangles
//   tile the polygon: non-zero areas, every triangle inside, no two overlapping, areas summing to
//   the polygon area.
// - The mapbox `earcut` package (3.2.4 in this repository's lockfile; it is a transitive
//   dependency, imported here as a test oracle only) triangulates the same float32 coordinates in
//   float64. Where its output is a tiling, the Earcut port must be valid, a tiling, and have the
//   same triangle count. Where it is not (some holes that touch at a vertex), the port must still
//   not return a wrong tiling as valid; since the review fixes of 2026-11-11 it tiles every polygon
//   of this search, so none may be flagged (finding frc-geometry-03 of
//   `fix-review-closing-geometry.md`; vertices inside edges are in
//   `gpu-polygon-triangulation-touching-rings.spec.ts`).
//
// The seeds named "hard" and the hole-bridge families failed on the kernel as it was before the
// closing pass of the math correctness review (2026-11-11): see
// `dev-docs/research/math-correctness-review-2026-11-10/core-numerics.md`.

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import earcut from 'earcut';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {checkTriangulation} from './polygon-triangulation-monotone-cases';
import {
  EARCUT_ONLY,
  MONOTONE_ONLY,
  triangulateBatch,
  type BatchResult
} from './polygon-triangulation-search-harness';
import {
  LATTICE_FRAME,
  METRE_FRAME,
  METRE_RESOLUTION_FRAME,
  createCloseCrossingPolygon,
  createCloseTangentPolygon,
  createNearEdgeHolePolygon,
  createRandomHoledPolygon,
  createSharedLeftmostPolygon,
  getPolygonDefect,
  getTilingDefect,
  mapToFrame,
  type RandomHoledPolygonOptions,
  type SearchFrame,
  type SearchPoint,
  type SearchPolygon
} from './polygon-triangulation-search-cases';

/** Earcut (the reference library, in float64) on the same float32 coordinates. */
function triangulateWithEarcut(rings: readonly SearchPoint[][]): number[][] {
  const coordinates = rings.flat().flat();
  const holeStarts: number[] = [];
  let row = 0;
  for (const ring of rings.slice(0, -1)) {
    row += ring.length;
    holeStarts.push(row);
  }
  const indices = earcut(coordinates, holeStarts);
  const triangles: number[][] = [];
  for (let at = 0; at < indices.length; at += 3) {
    triangles.push([indices[at], indices[at + 1], indices[at + 2]]);
  }
  return triangles;
}

type SearchCounts = {
  polygons: number;
  /** Polygons the earcut package does not tile. */
  referenceDefects: number;
  /** `valid = 0` results. */
  flagged: number;
};

/**
 * Runs the Earcut port on `polygons` and returns what departs from the reference (see the file
 * comment), one line per polygon, with the counts.
 */
async function getEarcutPathDepartures(
  device: Device,
  polygons: readonly SearchPolygon[]
): Promise<{departures: string[]; counts: SearchCounts; result: BatchResult}> {
  const result = await triangulateBatch(device, polygons, EARCUT_ONLY);
  const departures: string[] = [];
  const counts: SearchCounts = {polygons: polygons.length, referenceDefects: 0, flagged: 0};
  polygons.forEach((polygon, index) => {
    const reference = triangulateWithEarcut(polygon.rings);
    const referenceTiles = getTilingDefect(polygon.rings, reference) === undefined;
    const isValid = result.valid[index] === 1;
    const defect = isValid ? getTilingDefect(polygon.rings, result.triangles[index]) : undefined;
    counts.referenceDefects += referenceTiles ? 0 : 1;
    counts.flagged += isValid ? 0 : 1;
    if (defect) {
      departures.push(`${polygon.name}: valid = 1 and ${defect}`);
    } else if (referenceTiles && !isValid) {
      departures.push(`${polygon.name}: valid = 0 where earcut tiles the polygon`);
    } else if (referenceTiles && result.triangles[index].length !== reference.length) {
      departures.push(
        `${polygon.name}: ${result.triangles[index].length} triangles, earcut has ${reference.length}`
      );
    }
  });
  return {departures, counts, result};
}

function ringsTouch(rings: readonly SearchPoint[][]): boolean {
  const seen = new Set<string>();
  for (const ring of rings) {
    for (const point of ring) {
      const key = `${point[0]},${point[1]}`;
      if (seen.has(key)) {
        return true;
      }
      seen.add(key);
    }
  }
  return false;
}

/**
 * Runs the monotone path alone on `polygons`. Most polygons whose rings touch at a vertex are
 * flagged by that path, as documented (the caller's Earcut pass takes them), so for those only a
 * wrong tiling returned as valid is a departure. Every other polygon must be valid with exactly
 * `vertices - 2 + 2 holes` triangles that tile it; triangles of three collinear vertices have
 * zero area and are allowed, and the directed triangle edges must cancel to the boundary.
 */
async function getMonotonePathDepartures(
  device: Device,
  polygons: readonly SearchPolygon[]
): Promise<{departures: string[]; counts: SearchCounts}> {
  const result = await triangulateBatch(device, polygons, MONOTONE_ONLY);
  const departures: string[] = [];
  const counts: SearchCounts = {polygons: polygons.length, referenceDefects: 0, flagged: 0};
  polygons.forEach((polygon, index) => {
    const isValid = result.valid[index] === 1;
    const triangles = result.triangles[index];
    const defect = isValid ? getTilingDefect(polygon.rings, triangles, true) : undefined;
    counts.flagged += isValid ? 0 : 1;
    if (defect) {
      departures.push(`${polygon.name}: valid = 1 and ${defect}`);
      return;
    }
    if (ringsTouch(polygon.rings)) {
      return;
    }
    const vertexCount = polygon.rings.reduce((sum, ring) => sum + ring.length, 0);
    const expectedCount = vertexCount - 2 + 2 * (polygon.rings.length - 1);
    if (!isValid) {
      departures.push(`${polygon.name}: valid = 0`);
    } else if (triangles.length !== expectedCount) {
      departures.push(`${polygon.name}: ${triangles.length} triangles, expected ${expectedCount}`);
    } else if (
      !checkTriangulation(
        polygon.rings,
        triangles.map(triangle => [triangle[0], triangle[1], triangle[2]] as const)
      ).boundaryMatches
    ) {
      departures.push(`${polygon.name}: the triangle edges do not cancel to the boundary`);
    }
  });
  return {departures, counts};
}

type RandomFamily = 'two-opt' | 'many-holes' | 'star';

function getRandomOptions(
  family: RandomFamily,
  seed: number,
  touchProbability: number
): RandomHoledPolygonOptions {
  if (family === 'two-opt') {
    const dense = seed % 3 === 0;
    return {
      range: dense ? 14 + (seed % 7) : 40 + (seed % 5) * 40,
      shellVertexCount: 6 + (seed % 19),
      holeCount: 1 + (seed % 6),
      holeRadius: dense ? 3 : 4 + (seed % 4) * 6,
      touchProbability
    };
  }
  if (family === 'many-holes') {
    return {
      range: 50 + (seed % 4) * 20,
      shellVertexCount: 4 + (seed % 3),
      holeCount: 7 + (seed % 6),
      holeRadius: 3 + (seed % 4),
      touchProbability
    };
  }
  // More than 80 vertices: the z-order hash of the ear test is in use.
  return {
    shell: 'star',
    range: 600 + (seed % 3) * 200,
    shellVertexCount: 70 + (seed % 90),
    holeCount: 2 + (seed % 7),
    holeRadius: 10 + (seed % 3) * 15,
    touchProbability
  };
}

function createRandomPolygons(
  family: RandomFamily,
  frame: SearchFrame,
  seeds: readonly number[],
  touchProbability: number
): SearchPolygon[] {
  const polygons: SearchPolygon[] = [];
  for (const seed of seeds) {
    const rings = createRandomHoledPolygon(seed, getRandomOptions(family, seed, touchProbability));
    if (rings && rings.length > 1) {
      const name = `${family} ${frame.name} touch ${touchProbability} seed ${seed}`;
      expect(getPolygonDefect(rings, touchProbability > 0), name).toBeUndefined();
      polygons.push({name, rings: mapToFrame(frame, rings)});
    }
  }
  return polygons;
}

function getSeeds(count: number, hardSeeds: readonly number[] = []): number[] {
  return [...Array.from({length: count}, (_, index) => index + 1), ...hardSeeds];
}

// Seeds beyond the first ones that the search found to fail before the closing pass.
// 731 and 1475: two bridge candidates exactly collinear with the hole vertex, where the tie went
// to the farther one and the polygon came back `valid = 0`.
// 724, 1230, 1269, 1426 (float32-resolution frame): a wrong hit edge or tangent order.
const HARD_TWO_OPT_SEEDS = [724, 731, 1230, 1269, 1426, 1475];
// Holes that touch at a vertex, which the earcut package itself does not tile.
const REFERENCE_DEFECT_TWO_OPT_SEEDS = [633, 2446];
// A zero-length diagonal between two convex nodes at a vertex shared by two holes, which the
// exact midpoint test accepted: `valid = 0` where the earcut package tiles the polygon.
const ZERO_LENGTH_DIAGONAL_MANY_HOLES_SEED = 1936;
// Holes that do not touch, which the earcut package (3.2.4) does not tile and the port does.
const REFERENCE_DEFECT_MANY_HOLES_SEED = 2417;

/** The random configurations: family, frame, touch probability, seeds. */
const RANDOM_CONFIGURATIONS: [RandomFamily, SearchFrame, number, number[]][] = [
  ['two-opt', LATTICE_FRAME, 0, getSeeds(150, HARD_TWO_OPT_SEEDS)],
  ['two-opt', METRE_RESOLUTION_FRAME, 0, getSeeds(150, HARD_TWO_OPT_SEEDS)],
  ['two-opt', METRE_FRAME, 0, getSeeds(60, HARD_TWO_OPT_SEEDS)],
  ['two-opt', LATTICE_FRAME, 0.5, getSeeds(100, REFERENCE_DEFECT_TWO_OPT_SEEDS)],
  [
    'two-opt',
    METRE_RESOLUTION_FRAME,
    0.5,
    getSeeds(100, [...HARD_TWO_OPT_SEEDS, ...REFERENCE_DEFECT_TWO_OPT_SEEDS])
  ],
  ['many-holes', METRE_RESOLUTION_FRAME, 0, getSeeds(60, [REFERENCE_DEFECT_MANY_HOLES_SEED])],
  ['many-holes', LATTICE_FRAME, 0.5, getSeeds(60, [ZERO_LENGTH_DIAGONAL_MANY_HOLES_SEED])],
  ['star', METRE_RESOLUTION_FRAME, 0, getSeeds(30)],
  ['star', LATTICE_FRAME, 0.5, getSeeds(20)]
];

it('the Earcut port tiles random polygons with holes exactly and agrees with the earcut package', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  let polygonCount = 0;
  // Results on the integer lattice by family, touch probability and seed.
  const latticeResults = new Map<string, string>();
  const frameDepartures: string[] = [];
  for (const [family, frame, touchProbability, seeds] of RANDOM_CONFIGURATIONS) {
    const polygons = createRandomPolygons(family, frame, seeds, touchProbability);
    const {departures, counts, result} = await getEarcutPathDepartures(device, polygons);
    expect(departures, `${family} ${frame.name} touch ${touchProbability}`).toEqual([]);
    // Every polygon is valid, with touching rings or without.
    expect(counts.flagged, `${family} ${frame.name} touch ${touchProbability}`).toBe(0);
    polygonCount += counts.polygons;
    // Every decision but one is an exact predicate, so the indices do not change when the
    // lattice is moved to projected-metre magnitude. The exception is the float32 midpoint of a
    // candidate diagonal, which the 2 m lattice of METRE_FRAME represents exactly; on the
    // float32-resolution lattice it rounds, and about 4 polygons in 1,000 take another (valid)
    // diagonal, so that frame is not compared.
    polygons.forEach((polygon, index) => {
      const key = polygon.name.replace(` ${frame.name} `, ' ');
      const value = JSON.stringify([result.valid[index], result.triangles[index]]);
      if (frame === LATTICE_FRAME) {
        latticeResults.set(key, value);
      } else if (frame === METRE_FRAME && latticeResults.get(key) !== value) {
        frameDepartures.push(polygon.name);
      }
    });
  }
  expect(frameDepartures).toEqual([]);
  expect(polygonCount).toBeGreaterThan(700);
}, 300_000);

it('the monotone path alone tiles the same polygons with vertices - 2 + 2 holes triangles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [family, frame, touchProbability, seeds] of RANDOM_CONFIGURATIONS) {
    const polygons = createRandomPolygons(family, frame, seeds, touchProbability);
    const {departures, counts} = await getMonotonePathDepartures(device, polygons);
    expect(departures, `${family} ${frame.name} touch ${touchProbability}`).toEqual([]);
    if (touchProbability === 0) {
      expect(counts.flagged).toBe(0);
    }
  }
}, 300_000);

it('the hole bridge does not depend on float32 rounding of the ray crossing or of the tangents', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons: SearchPolygon[] = [];
  for (const frame of [LATTICE_FRAME, METRE_RESOLUTION_FRAME]) {
    for (let seed = 1; seed <= 12; seed++) {
      // Two shell edges crossed by the ray less than one float32 step apart, in every order in
      // which the walk can meet them.
      for (let rotation = 0; rotation < 7; rotation++) {
        polygons.push({
          name: `close crossing ${frame.name} seed ${seed} rotation ${rotation}`,
          rings: mapToFrame(frame, createCloseCrossingPolygon(seed, rotation))
        });
      }
    }
    for (let seed = 1; seed <= 40; seed++) {
      for (const flag of [false, true]) {
        // Two bridge candidates whose tangents a float32 division cannot tell apart.
        polygons.push({
          name: `close tangent ${frame.name} seed ${seed} mirror ${flag}`,
          rings: mapToFrame(frame, createCloseTangentPolygon(seed, flag))
        });
        // Two holes that share their leftmost vertex with first edges of almost one slope.
        polygons.push({
          name: `shared leftmost ${frame.name} seed ${seed} upper first ${flag}`,
          rings: mapToFrame(frame, createSharedLeftmostPolygon(seed, flag))
        });
      }
    }
  }
  // A hole vertex a few float32 steps right of a slanted shell edge, in a frame that spans zero
  // (the vertex is far finer than the edge's ends) and at projected-metre magnitude.
  for (const frame of ['local', 'metre'] as const) {
    let count = 0;
    for (let seed = 1; count < 400; seed++) {
      const rings = createNearEdgeHolePolygon(seed, frame);
      if (rings) {
        polygons.push({name: `near edge ${frame} seed ${seed}`, rings});
        count++;
      }
    }
  }
  for (const polygon of polygons) {
    expect(getPolygonDefect(polygon.rings, true), polygon.name).toBeUndefined();
  }
  const {departures, counts} = await getEarcutPathDepartures(device, polygons);
  expect(departures).toEqual([]);
  expect(counts).toEqual({polygons: polygons.length, referenceDefects: 0, flagged: 0});
}, 300_000);
