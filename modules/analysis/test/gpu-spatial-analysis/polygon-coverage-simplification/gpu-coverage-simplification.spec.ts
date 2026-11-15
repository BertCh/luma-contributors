// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getGPULineSimplificationParameterValues} from '../../../src/gpu-spatial-analysis/line-simplification/index';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  createSeededRandom,
  flattenPolygons,
  type OraclePolygons
} from '../spatial-weights/spatial-weights-oracle';
import {
  computeCoverageSimplificationOracle,
  countCoverageCrossings,
  findCoverageGaps
} from './coverage-simplification-oracle';
import {createWavyStrips} from './coverage-simplification-scenes';
import {requireTestDevice} from '../../utils/require-test-device';

type Point = [number, number];

/**
 * A jagged coverage: a grid of cells whose sides are wiggled polylines. Neighbours share the
 * identical side points, so the coverage is gap-free before simplification.
 */
function createJaggedGrid(columns: number, rows: number, seed: number): OraclePolygons {
  const random = createSeededRandom(seed);
  const sideCache = new Map<string, Point[]>();
  // A side from corner a to corner b, stored in canonical direction (lower corner first).
  const side = (ax: number, ay: number, bx: number, by: number): Point[] => {
    const forward = ax < bx || (ax === bx && ay < by);
    const [x0, y0, x1, y1] = forward ? [ax, ay, bx, by] : [bx, by, ax, ay];
    const key = `${x0},${y0},${x1},${y1}`;
    if (!sideCache.has(key)) {
      const points: Point[] = [[x0, y0]];
      const steps = 12;
      const border =
        (x0 === 0 && x1 === 0) ||
        (y0 === 0 && y1 === 0) ||
        (x0 === columns && x1 === columns) ||
        (y0 === rows && y1 === rows);
      for (let step = 1; step < steps; step++) {
        const t = step / steps;
        const wiggle = border ? 0 : (random() - 0.5) * 0.3;
        points.push(
          x0 === x1 ? [x0 + wiggle, y0 + (y1 - y0) * t] : [x0 + (x1 - x0) * t, y0 + wiggle]
        );
      }
      points.push([x1, y1]);
      sideCache.set(key, points);
    }
    const points = sideCache.get(key)!;
    return forward ? points : [...points].reverse();
  };
  const polygons: OraclePolygons = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      let ring: Point[] = [
        ...side(x, y, x + 1, y).slice(0, -1),
        ...side(x + 1, y, x + 1, y + 1).slice(0, -1),
        ...side(x + 1, y + 1, x, y + 1).slice(0, -1),
        ...side(x, y + 1, x, y).slice(0, -1)
      ];
      const rotation = Math.floor(random() * ring.length);
      ring = [...ring.slice(rotation), ...ring.slice(0, rotation)];
      if (random() < 0.4) ring.reverse();
      polygons.push([ring]);
    }
  }
  return polygons;
}

/** A square frame with a wiggly square hole, and an island that fills the hole exactly. */
function createHoleAndIsland(islandFirst: boolean): OraclePolygons {
  const outer: Point[] = [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10]
  ];
  const holeRing: Point[] = [];
  const corners: Point[] = [
    [3, 3],
    [3, 7],
    [7, 7],
    [7, 3]
  ];
  for (let corner = 0; corner < 4; corner++) {
    const [ax, ay] = corners[corner];
    const [bx, by] = corners[(corner + 1) % 4];
    for (let step = 0; step < 8; step++) {
      const t = step / 8;
      const bump = step === 0 ? 0 : (step % 2 ? 0.1 : -0.1) * (corner % 2 ? 1 : -1);
      holeRing.push(ax === bx ? [ax + bump, ay + (by - ay) * t] : [ax + (bx - ax) * t, ay + bump]);
    }
  }
  const island = [...holeRing].reverse();
  const frame = [outer, holeRing];
  return islandFirst ? [[island], frame] : [frame, [island]];
}

type Result = {
  keepMask: number[];
  positions: number[];
  ringOffsets: number[];
  overflow: number;
  requiredCount: number;
  uncertainCount: number;
  converged: number;
  /** `[initial crossings, remaining crossings, restored vertices, overflow]`. */
  topologyStats: number[];
};

async function runCoverage(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  polygons: OraclePolygons,
  tolerance: number,
  capacity?: number,
  topologyRounds = 0,
  simplifyBoundary = true,
  padding?: {value: number; leading: number; trailing: number; snapTolerance: number}
): Promise<Result> {
  const layout = flattenPolygons(polygons);
  if (padding) {
    const positions = new Float32Array(
      layout.positions.length + 2 * (padding.leading + padding.trailing)
    ).fill(padding.value);
    positions.set(layout.positions, 2 * padding.leading);
    layout.positions = positions;
    layout.ringOffsets = Uint32Array.from(layout.ringOffsets, offset => offset + padding.leading);
  }
  const vertexCount = layout.positions.length / 2;
  const ringCount = layout.ringOffsets.length - 1;
  const rig = new WeightsRig(device);
  const outputPositions = rig.output('float32x2', capacity ?? vertexCount);
  const outputOffsets = rig.output('uint32', ringCount + 1);
  const keepMask = rig.output('uint32', vertexCount);
  const overflow = rig.output('uint32', 1);
  const requiredCount = rig.output('uint32', 1);
  const converged = rig.output('uint32', 1);
  const uncertainCount = rig.output('uint32', 1);
  const topologyStats = rig.output('uint32', 4);
  const parameters = new GPUParameterBuffer(device, {
    id: 'coverage-tolerance',
    format: 'float32',
    length: 4,
    values: getGPULineSimplificationParameterValues({tolerance})
  });
  rig.run(
    new GPUCoverageSimplification({
      positions: rig.input(layout.positions, 'float32x2', vertexCount),
      ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
      polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
      featureOffsets: rig.input(
        Uint32Array.from(layout.polygonOffsets, (_, index) => index),
        'uint32',
        layout.polygonOffsets.length
      ),
      parameters: parameters.importToGraph(rig.graph),
      snapTolerance: padding?.snapTolerance,
      maximumIterations: 128,
      simplifyBoundary,
      topologyIterationCount: topologyRounds,
      output: {
        positions: outputPositions.view,
        ringOffsets: outputOffsets.view,
        outputMask: keepMask.view,
        topologyStatistics: topologyStats.view,
        status: {
          converged: converged.view,
          overflow: overflow.view,
          requiredCount: requiredCount.view,
          uncertainCount: uncertainCount.view
        }
      }
    })
  );
  const ringOffsetValues = await readUint32(outputOffsets.buffer, ringCount + 1);
  const result = {
    keepMask: await readUint32(keepMask.buffer, vertexCount),
    positions: await readFloat32(outputPositions.buffer, 2 * ringOffsetValues[ringCount]),
    ringOffsets: ringOffsetValues,
    overflow: (await readUint32(overflow.buffer, 1))[0],
    requiredCount: (await readUint32(requiredCount.buffer, 1))[0],
    uncertainCount: (await readUint32(uncertainCount.buffer, 1))[0],
    converged: (await readUint32(converged.buffer, 1))[0],
    topologyStats: Array.from(await readUint32(topologyStats.buffer, 4))
  };
  parameters.destroy();
  rig.destroy();
  return result;
}

function expectMatchesOracle(polygons: OraclePolygons, tolerance: number, result: Result): number {
  const layout = flattenPolygons(polygons);
  const expected = computeCoverageSimplificationOracle(polygons, tolerance);
  expect(result.keepMask).toEqual(expected.keepMask);
  const keptTotal = expected.keepMask.reduce((sum, flag) => sum + flag, 0);
  expect(result.requiredCount).toBe(keptTotal);
  expect(result.overflow).toBe(0);
  expect(result.converged).toBe(1);
  // Ring offsets and positions are the filtered input, ring after ring.
  const ringCount = layout.ringOffsets.length - 1;
  let rank = 0;
  for (let ring = 0; ring < ringCount; ring++) {
    expect(result.ringOffsets[ring]).toBe(rank);
    for (let vertex = layout.ringOffsets[ring]; vertex < layout.ringOffsets[ring + 1]; vertex++) {
      if (expected.keepMask[vertex]) {
        expect(result.positions[2 * rank]).toBe(layout.positions[2 * vertex]);
        expect(result.positions[2 * rank + 1]).toBe(layout.positions[2 * vertex + 1]);
        rank++;
      }
    }
  }
  expect(result.ringOffsets[ringCount]).toBe(rank);
  return keptTotal;
}

it('GPUCoverageSimplification matches the oracle and stays gap-free on a jagged grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const seed of [1, 2]) {
    const polygons = createJaggedGrid(6, 4, seed);
    const vertexCount = flattenPolygons(polygons).positions.length / 2;
    const kept = new Map<number, number>();
    for (const tolerance of [0, 0.04, 0.12]) {
      const result = await runCoverage(device, polygons, tolerance);
      kept.set(tolerance, expectMatchesOracle(polygons, tolerance, result));
      expect(findCoverageGaps(polygons, result.keepMask), `seed ${seed} tol ${tolerance}`).toEqual(
        []
      );
    }
    // Tolerance 0 keeps every non-collinear vertex, larger tolerances remove vertices, never all.
    expect(kept.get(0)!).toBeGreaterThan(vertexCount * 0.7);
    expect(kept.get(0.04)!).toBeLessThan(kept.get(0)!);
    expect(kept.get(0.12)!).toBeLessThan(kept.get(0.04)!);
    expect(kept.get(0.12)!).toBeGreaterThan(24);
  }
});

it('GPUCoverageSimplification handles holes and islands, either owner', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const islandFirst of [false, true]) {
    const polygons = createHoleAndIsland(islandFirst);
    const vertexCount = flattenPolygons(polygons).positions.length / 2;
    const result = await runCoverage(device, polygons, 0.15);
    const kept = expectMatchesOracle(polygons, 0.15, result);
    expect(kept).toBeLessThan(vertexCount);
    expect(kept).toBeGreaterThan(8);
    expect(findCoverageGaps(polygons, result.keepMask)).toEqual([]);
  }
});

it('GPUCoverageSimplification clamps to the output capacity and flags overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createJaggedGrid(3, 3, 5);
  const full = await runCoverage(device, polygons, 0);
  const capacity = Math.floor(full.requiredCount / 2);
  const small = await runCoverage(device, polygons, 0, capacity);
  expect(small.overflow).toBe(1);
  expect(small.requiredCount).toBe(full.requiredCount);
  expect(small.ringOffsets.every(offset => offset <= capacity)).toBe(true);
  expect(small.ringOffsets.at(-1)).toBe(capacity);
});

it('GPUCoverageSimplification copies long arcs, shared and isolated, in parallel', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(11);
  const arc: Point[] = [];
  for (let i = 0; i < 20000; i++) {
    arc.push([
      Math.fround(i / 19999),
      Math.fround(0.02 * Math.sin(i * 0.002) + (random() - 0.5) * 0.0004)
    ]);
  }
  const ring: Point[] = [];
  for (let i = 0; i < 20000; i++) {
    const angle = (2 * Math.PI * i) / 20000;
    const radius = 3 + 0.01 * Math.sin(i * 0.01) + (random() - 0.5) * 0.0004;
    ring.push([Math.fround(10 + radius * Math.cos(angle)), Math.fround(radius * Math.sin(angle))]);
  }
  // Two polygons share the 20k-vertex arc; a third ring has no partner and no junction.
  const polygons: OraclePolygons = [
    [[...arc, [1, -1], [0, -1]]],
    [[...[...arc].reverse(), [0, 1], [1, 1]]],
    [ring]
  ];
  const result = await runCoverage(device, polygons, 0.001);
  expectMatchesOracle(polygons, 0.001, result);
});

it('GPUCoverageSimplification repairs crossings of simplified arcs and keeps rings valid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createWavyStrips(5, 1);
  const layout = flattenPolygons(polygons);
  const vertexCount = layout.positions.length / 2;
  expect(countCoverageCrossings(layout.positions, layout.ringOffsets)).toBe(0);
  // Plain Douglas-Peucker on each shared arc makes neighbouring boundaries cross.
  const plain = await runCoverage(device, polygons, 0.1, undefined, 0);
  expect(
    countCoverageCrossings(plain.positions, plain.ringOffsets),
    'plain Douglas-Peucker crosses'
  ).toBeGreaterThan(0);
  expect(plain.topologyStats).toEqual([0, 0, 0, 0]);

  const repaired = await runCoverage(device, polygons, 0.1, undefined, 12);
  const [initial, remaining, restored, overflow] = repaired.topologyStats;
  expect(overflow).toBe(0);
  expect(initial).toBeGreaterThan(0);
  expect(remaining).toBe(0);
  expect(restored).toBeGreaterThan(0);
  expect(countCoverageCrossings(repaired.positions, repaired.ringOffsets)).toBe(0);
  expect(findCoverageGaps(polygons, repaired.keepMask)).toEqual([]);
  // Repair only adds vertices to the plain result, and keeps simplifying.
  const kept = (result: Result) => result.keepMask.reduce((sum, flag) => sum + flag, 0);
  expect(kept(repaired)).toBeGreaterThanOrEqual(kept(plain));
  expect(kept(repaired)).toBeLessThan(vertexCount);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    if (plain.keepMask[vertex]) expect(repaired.keepMask[vertex]).toBe(1);
  }
  // Pinned against Shapely 2.1.2 on this scene: the input is a valid coverage, the plain result is
  // not (3 invalid polygons, coverage_is_valid false), and the repaired result passes
  // shapely.is_valid for every polygon and shapely.coverage_is_valid (no gaps, no overlaps).
  // Shapely's own coverage_simplify keeps 212 vertices here (area-based), the repair keeps 118.
  // The first count is 12 crossing pairs plus 5 spans whose shortcut passes over a kept vertex of
  // a neighbour (the sweep test, review 2026-11-10 polygon-topology-02; it was 12 before that
  // test existed). The restored and kept counts did not change, so on this scene the sweep test
  // restores no vertex that the crossing repair had not.
  expect(repaired.topologyStats).toEqual([17, 0, 18, 0]);
  expect(kept(repaired)).toBe(118);
  // The strip corners are nodes of the coverage and are always kept.
  const nodeVertices = polygons.flatMap(([ring]) =>
    ring.flatMap(([x], vertex) => (x === 0 || x === 10 ? [vertex] : []))
  );
  expect(nodeVertices.length).toBeGreaterThan(0);
});

it('GPUCoverageSimplification never collapses a ring below three vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createJaggedGrid(3, 3, 9);
  // A tolerance larger than every cell collapses plain Douglas-Peucker rings.
  const result = await runCoverage(device, polygons, 5, undefined, 4);
  const ringCount = result.ringOffsets.length - 1;
  for (let ring = 0; ring < ringCount; ring++) {
    expect(result.ringOffsets[ring + 1] - result.ringOffsets[ring]).toBeGreaterThanOrEqual(3);
  }
  expect(result.topologyStats[3]).toBe(0);
  expect(countCoverageCrossings(result.positions, result.ringOffsets)).toBe(0);
});

// Generated by shapely 2.1.2: `coverage_simplify([A, B], 0.5, simplify_boundary=...)`. A and B share
// the edge x = 10 (a big bump and two wiggles); both also have wiggles on the outer boundary,
// (-0.02, 5) on A and (14, -0.03) on B, that only `simplify_boundary=True` removes.
const SHAPELY_COVERAGE: OraclePolygons = [
  [
    [
      [0.0, 0.0],
      [10.0, 0.0],
      [10.02, 2.0],
      [10.0, 3.0],
      [12.5, 5.0],
      [10.0, 7.0],
      [10.03, 8.5],
      [10.0, 10.0],
      [7.0, 10.03],
      [5.0, 12.5],
      [3.0, 10.02],
      [0.0, 10.0],
      [-0.02, 5.0]
    ]
  ],
  [
    [
      [10.0, 0.0],
      [14.0, -0.03],
      [20.0, 0.0],
      [20.0, 10.0],
      [15.0, 13.0],
      [13.0, 10.02],
      [10.0, 10.0],
      [10.03, 8.5],
      [10.0, 7.0],
      [12.5, 5.0],
      [10.0, 3.0],
      [10.02, 2.0]
    ]
  ]
];
const SHAPELY_COVERAGE_KEPT: Record<'true' | 'false', Point[][]> = {
  true: [
    [
      [10.0, 0.0],
      [10.0, 3.0],
      [12.5, 5.0],
      [10.0, 7.0],
      [10.0, 10.0],
      [7.0, 10.03],
      [5.0, 12.5],
      [3.0, 10.02],
      [0.0, 10.0],
      [0.0, 0.0]
    ],
    [
      [10.0, 0.0],
      [20.0, 0.0],
      [20.0, 10.0],
      [15.0, 13.0],
      [13.0, 10.02],
      [10.0, 10.0],
      [10.0, 7.0],
      [12.5, 5.0],
      [10.0, 3.0]
    ]
  ],
  false: [
    [
      [10.0, 0.0],
      [10.0, 3.0],
      [12.5, 5.0],
      [10.0, 7.0],
      [10.0, 10.0],
      [7.0, 10.03],
      [5.0, 12.5],
      [3.0, 10.02],
      [0.0, 10.0],
      [-0.02, 5.0],
      [0.0, 0.0]
    ],
    [
      [10.0, 0.0],
      [14.0, -0.03],
      [20.0, 0.0],
      [20.0, 10.0],
      [15.0, 13.0],
      [13.0, 10.02],
      [10.0, 10.0],
      [10.0, 7.0],
      [12.5, 5.0],
      [10.0, 3.0]
    ]
  ]
};

function getKeptRings(polygons: OraclePolygons, result: Result): Point[][] {
  const layout = flattenPolygons(polygons);
  const rings: Point[][] = [];
  for (let ring = 0; ring < layout.ringOffsets.length - 1; ring++) {
    const kept: Point[] = [];
    for (let row = result.ringOffsets[ring]; row < result.ringOffsets[ring + 1]; row++) {
      kept.push([result.positions[2 * row], result.positions[2 * row + 1]]);
    }
    rings.push(kept);
  }
  return rings;
}

const sortPoints = (points: Point[]) =>
  points
    .map(([x, y]) => [Math.fround(x), Math.fround(y)])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);

it('GPUCoverageSimplification simplifyBoundary matches shapely coverage_simplify', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const simplifyBoundary of [true, false]) {
    const result = await runCoverage(device, SHAPELY_COVERAGE, 0.5, undefined, 4, simplifyBoundary);
    const expected = SHAPELY_COVERAGE_KEPT[String(simplifyBoundary) as 'true' | 'false'];
    const rings = getKeptRings(SHAPELY_COVERAGE, result);
    // Vertex sets per polygon (shapely may start a ring elsewhere).
    expect(rings.map(sortPoints), String(simplifyBoundary)).toEqual(expected.map(sortPoints));
  }
});

it('GPUCoverageSimplification simplifyBoundary false keeps every unshared vertex', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const columns = 6;
  const rows = 4;
  const polygons = createJaggedGrid(columns, rows, 1);
  const layout = flattenPolygons(polygons);
  const vertexCount = layout.positions.length / 2;
  const simplified = await runCoverage(device, polygons, 0.12);
  const preserved = await runCoverage(device, polygons, 0.12, undefined, 0, false);
  let boundaryCount = 0;
  let boundaryDropped = 0;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const x = layout.positions[2 * vertex];
    const y = layout.positions[2 * vertex + 1];
    // The jagged grid only wiggles interior sides, so the outer boundary lies on the frame.
    if (x === 0 || x === columns || y === 0 || y === rows) {
      boundaryCount++;
      boundaryDropped += simplified.keepMask[vertex] === 0 ? 1 : 0;
      expect(preserved.keepMask[vertex]).toBe(1);
    }
    // Interior decisions are unchanged, and nothing is dropped that default mode kept.
    expect(preserved.keepMask[vertex]).toBeGreaterThanOrEqual(simplified.keepMask[vertex]);
  }
  expect(boundaryCount).toBeGreaterThan(0);
  expect(boundaryDropped).toBeGreaterThan(0);
});

/** Even-odd containment of a point in a ring, float64 (the test points are far from every edge). */
function isInsideRing(point: Point, ring: readonly Point[]): boolean {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [ax, ay] = ring[previous];
    const [bx, by] = ring[index];
    if (ay > point[1] !== by > point[1]) {
      if (point[0] < ((bx - ax) * (point[1] - ay)) / (by - ay) + ax) {
        inside = !inside;
      }
    }
  }
  return inside;
}

function getOutputRings(result: Result): Point[][] {
  const rings: Point[][] = [];
  for (let ring = 0; ring + 1 < result.ringOffsets.length; ring++) {
    const vertices: Point[] = [];
    for (let vertex = result.ringOffsets[ring]; vertex < result.ringOffsets[ring + 1]; vertex++) {
      vertices.push([result.positions[2 * vertex], result.positions[2 * vertex + 1]]);
    }
    rings.push(vertices);
  }
  return rings;
}

/**
 * The scene of math correctness review 2026-11-10, polygon-topology-02. Polygons A (left, with a
 * triangular bulge to (3, 5)) and B (right) share the arc (0,0), (0,4), (3,5), (0,6), (0,10). A
 * small island C sits inside the bulge, so A has a hole there.
 */
function createSweptIsland(): OraclePolygons {
  const island: Point[] = [
    [0.5, 4.8],
    [1.5, 5],
    [0.5, 5.2]
  ];
  const left: Point[] = [
    [-10, 0],
    [0, 0],
    [0, 4],
    [3, 5],
    [0, 6],
    [0, 10],
    [-10, 10]
  ];
  const right: Point[] = [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
    [0, 6],
    [3, 5],
    [0, 4]
  ];
  return [[left, [...island].reverse()], [right], [island]];
}

it('GPUCoverageSimplification does not sweep an arc over an island (review 02)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createSweptIsland();
  // Douglas-Peucker at tolerance 3.5 straightens the shared arc to x = 0: the farthest vertex,
  // (3, 5), is 3 units from the chord. No simplified segment crosses another, yet the island
  // would then lie in B while A keeps its hole outside its own shell. A valid result keeps every
  // island vertex inside A's shell and outside B's.
  const result = await runCoverage(device, polygons, 3.5, undefined, 4);
  expect(result.overflow).toBe(0);
  const [leftShell, leftHole, rightShell, island] = getOutputRings(result);
  expect(island.length).toBe(3);
  expect(leftHole.length).toBe(3);
  for (const vertex of island) {
    expect(isInsideRing(vertex, leftShell), `island vertex ${vertex} in A`).toBe(true);
    expect(isInsideRing(vertex, rightShell), `island vertex ${vertex} in B`).toBe(false);
  }
  // The sweep is reported with the crossings: one before repair, none left, one vertex restored.
  expect(result.topologyStats[0]).toBe(1);
  expect(result.topologyStats[1]).toBe(0);
  expect(result.topologyStats[2]).toBeGreaterThanOrEqual(1);
  // The shared arc is still simplified: the bulge keeps its tip and drops (0, 4) and (0, 6).
  expect(leftShell).toEqual([
    [-10, 0],
    [0, 0],
    [3, 5],
    [0, 10],
    [-10, 10]
  ]);
  expect(rightShell).toEqual([
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
    [3, 5]
  ]);
  // Without the island the same tolerance straightens the arc, as before.
  const plain = await runCoverage(device, [[polygons[0][0]], polygons[1]], 3.5, undefined, 4);
  const [plainLeft] = getOutputRings(plain);
  expect(plainLeft).toEqual([
    [-10, 0],
    [0, 0],
    [0, 10],
    [-10, 10]
  ]);
  expect(plain.topologyStats.slice(0, 3)).toEqual([0, 0, 0]);
});

/** Two squares of the given side, side by side at `northing`, with a vertex on the shared side. */
async function runProjectedSquares(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  side: number,
  northing: number,
  snapTolerance: number
) {
  const left: Point[] = [
    [0, northing],
    [side, northing],
    [side, northing + side / 2],
    [side, northing + side],
    [0, northing + side]
  ];
  const right: Point[] = [
    [side, northing],
    [2 * side, northing],
    [2 * side, northing + side],
    [side, northing + side],
    [side, northing + side / 2]
  ];
  const layout = flattenPolygons([[left], [right]]);
  const vertexCount = layout.positions.length / 2;
  const rig = new WeightsRig(device);
  const outputPositions = rig.output('float32x2', vertexCount);
  const outputOffsets = rig.output('uint32', 3);
  const overflow = rig.output('uint32', 1);
  const uncertainCount = rig.output('uint32', 1);
  const parameters = new GPUParameterBuffer(device, {
    id: 'coverage-tolerance-projected',
    format: 'float32',
    length: 4,
    values: getGPULineSimplificationParameterValues({tolerance: 1})
  });
  rig.run(
    new GPUCoverageSimplification({
      positions: rig.input(layout.positions, 'float32x2', vertexCount),
      ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
      polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
      featureOffsets: rig.input(layout.featureOffsets, 'uint32', layout.featureOffsets.length),
      parameters: parameters.importToGraph(rig.graph),
      snapTolerance,
      output: {
        positions: outputPositions.view,
        ringOffsets: outputOffsets.view,
        status: {overflow: overflow.view, uncertainCount: uncertainCount.view}
      }
    })
  );
  const result = {
    offsets: Array.from(await readUint32(outputOffsets.buffer, 3)),
    coordinates: await readFloat32(outputPositions.buffer, 2 * vertexCount),
    overflow: (await readUint32(overflow.buffer, 1))[0],
    uncertainCount: (await readUint32(uncertainCount.buffer, 1))[0]
  };
  parameters.destroy();
  rig.destroy();
  return result;
}

it('GPUCoverageSimplification snap keys follow the extent at projected coordinates (review 03)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two 100 m squares side by side at a northing of 3,000,000 m, with an extra vertex on the shared
  // side, 1 mm snap tolerance. Absolute snap keys gave both northings one key, so every vertex of
  // a side was one point and no edge survived.
  const northing = 3000000;
  const run = await runProjectedSquares(device, 100, northing, 0.001);
  expect(run.overflow).toBe(0);
  expect(run.uncertainCount).toBe(0);
  // Each square keeps its four corners; the collinear vertex on the shared side is dropped.
  expect(run.offsets).toEqual([0, 4, 8]);
  const kept: Point[] = [];
  for (let vertex = 0; vertex < 8; vertex++) {
    kept.push([run.coordinates[2 * vertex], run.coordinates[2 * vertex + 1] - northing]);
  }
  expect(kept).toEqual([
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100],
    [100, 0],
    [200, 0],
    [200, 100],
    [100, 100]
  ]);
  // Squares of 10,000 m: the two vertices at x = 20000 are 2e7 steps of 1 mm from the first
  // vertex, past the 2^24 steps float32 can key exactly. They are counted in `uncertainCount`;
  // `overflow` is for dropped rows only (fix review fr-geometry-07).
  const wide = await runProjectedSquares(device, 10000, northing, 0.001);
  expect(wide.uncertainCount).toBe(2);
  expect(wide.overflow).toBe(0);
  // Without a snap tolerance there is no range.
  const exact = await runProjectedSquares(device, 10000, northing, 0);
  expect(exact.uncertainCount).toBe(0);
});

it('GPUCoverageSimplification ignores leading and trailing garbage when choosing the snap origin', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons: OraclePolygons = [
    [
      [
        [1e6, 1e6],
        [1e6 + 1, 1e6],
        [1e6 + 1, 1e6 + 1],
        [1e6, 1e6 + 1]
      ]
    ],
    [
      [
        [1e6 + 1, 1e6],
        [1e6 + 2, 1e6],
        [1e6 + 2, 1e6 + 1],
        [1e6 + 1, 1e6 + 1]
      ]
    ]
  ];
  const snapTolerance = 0.001;
  const reference = await runCoverage(device, polygons, 0, 16, 0, false, {
    leading: 0,
    trailing: 0,
    value: 0,
    snapTolerance
  });
  expect(reference.uncertainCount).toBe(0);
  expect(reference.requiredCount).toBe(8);
  expect(reference.overflow).toBe(0);
  for (const value of [Number.NaN, 3e20, -3e20]) {
    const result = await runCoverage(device, polygons, 0, 16, 0, false, {
      leading: 2,
      trailing: 3,
      value,
      snapTolerance
    });
    expect(result.keepMask.slice(0, 2)).toEqual([0, 0]);
    expect(result.keepMask.slice(-3)).toEqual([0, 0, 0]);
    expect({...result, keepMask: result.keepMask.slice(2, -3)}).toEqual(reference);
  }
});
