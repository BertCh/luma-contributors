// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction';
import {createOptimizationHarness} from '../../gpu-network/network-optimization/optimization-test-harness';
import {requireTestDevice} from '../../utils/require-test-device';

type BoundaryCase = {
  name: string;
  /** Ring vertices `[x, y]`, implicitly closed. */
  ring: number[][];
  /** The two fixes of the track, at times 0 and 10. */
  track: [number[], number[]];
  /** Expected `[ever, always]` for closed and for open geometry. */
  closed: [number, number];
  open: [number, number];
};

// Review finding trajectories-lines-09. Coordinates are metres; every edge below is 100 km or
// longer, and every track runs parallel to an edge. The expected relationships are read off the
// geometry: a track 5 cm off an edge is strictly on one side of it, whatever the boundary rule,
// and only a track exactly on the edge depends on the rule. A tolerance band of
// 1e-6 * |edge| (10 cm here) called all of them "on the boundary".
const SQUARE = [
  [0, 0],
  [100_000, 0],
  [100_000, 100_000],
  [0, 100_000]
];
// Interior above the diagonal y = x. 0.0703125 is 18 float32 steps at 50 km, 4.97 cm from the line.
const TRIANGLE = [
  [0, 0],
  [100_000, 100_000],
  [0, 100_000]
];
const CASES: BoundaryCase[] = [
  {
    name: '5 cm outside a 100 km axis-aligned edge',
    ring: SQUARE,
    track: [
      [50_000, -0.05],
      [60_000, -0.05]
    ],
    closed: [0, 0],
    open: [0, 0]
  },
  {
    name: '5 cm inside a 100 km axis-aligned edge',
    ring: SQUARE,
    track: [
      [50_000, 0.05],
      [60_000, 0.05]
    ],
    closed: [1, 1],
    open: [1, 1]
  },
  {
    name: 'exactly on an axis-aligned edge',
    ring: SQUARE,
    track: [
      [50_000, 0],
      [60_000, 0]
    ],
    closed: [1, 1],
    open: [0, 0]
  },
  {
    name: '5 cm outside a 141 km diagonal edge',
    ring: TRIANGLE,
    track: [
      [50_000, 50_000 - 0.0703125],
      [60_000, 60_000 - 0.0703125]
    ],
    closed: [0, 0],
    open: [0, 0]
  },
  {
    name: '5 cm inside a 141 km diagonal edge',
    ring: TRIANGLE,
    track: [
      [50_000, 50_000 + 0.0703125],
      [60_000, 60_000 + 0.0703125]
    ],
    closed: [1, 1],
    open: [1, 1]
  },
  {
    name: 'exactly on a diagonal edge',
    ring: TRIANGLE,
    track: [
      [50_000, 50_000],
      [60_000, 60_000]
    ],
    closed: [1, 1],
    open: [0, 0]
  }
];

it('GPUTrajectoryRestriction crosses a 128-vertex ring through two of its vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A regular 128-gon of radius 10 (the largest ring the contributor takes), with vertex 0 at
  // (10, 0) and vertex 64 at (-10, 0) exactly. The track runs along y = 0 from x = -20 to x = 20
  // in 40 time units, so it enters through vertex 64 at time 10 and leaves through vertex 0 at
  // time 30. Each of those crossings is found on two edges, with parameters that can differ by
  // float32 rounding; the pieces between them have no interior.
  const vertexCount = 128;
  const ring = Array.from({length: vertexCount}, (_, vertex) => {
    const angle = (2 * Math.PI * vertex) / vertexCount;
    return [10 * Math.cos(angle), 10 * Math.sin(angle)];
  });
  ring[0] = [10, 0];
  ring[64] = [-10, 0];
  const slots = 8;
  for (const geometryBoundary of ['closed', 'open'] as const) {
    const harness = createOptimizationHarness(device);
    const output = {
      intervals: harness.output('float32x2', 2 * slots),
      positions: harness.output('float32x4', 2 * slots),
      status: harness.output('uint32', 2 * slots),
      relationships: harness.output('uint32x4', 1)
    };
    const result = await harness.run(
      new GPUTrajectoryRestriction({
        positions: harness.input('float32x2', Float32Array.of(-20, 0, 20, 0)),
        times: harness.input('float32', Float32Array.of(0, 40)),
        offsets: harness.input('uint32', Uint32Array.of(0, 2)),
        parameters: harness.input(
          'float32',
          Float32Array.from([0, 40, 0, 0, 0, 0, 0, ...ring.flat()])
        ),
        polygonRingVertexCounts: [vertexCount],
        intervalsPerSegment: slots,
        geometryBoundary,
        output
      })
    );
    const intervals = result(output.intervals);
    const status = result(output.status);
    const pieces: number[][] = [];
    for (let slot = 0; slot < slots; slot++) {
      if ((status[slot] & 3) !== 0) {
        pieces.push([intervals[2 * slot], intervals[2 * slot + 1]]);
      }
    }
    expect(status[0] & 256, geometryBoundary).toBe(0);
    expect(pieces.length, geometryBoundary).toBeGreaterThan(0);
    // The pieces are in time order, touch each other, and together span [10, 30].
    expect(pieces[0][0], geometryBoundary).toBeCloseTo(10, 4);
    expect(pieces[pieces.length - 1][1], geometryBoundary).toBeCloseTo(30, 4);
    for (let piece = 1; piece < pieces.length; piece++) {
      expect(pieces[piece][0], geometryBoundary).toBe(pieces[piece - 1][1]);
    }
    // Open geometry: one piece, the interior only. Closed: slivers at the two vertices may add
    // zero-width pieces, never a gap.
    if (geometryBoundary === 'open') {
      expect(pieces.length).toBe(1);
      expect(status[0] & 192).toBe(192);
    }
    expect(result(output.relationships).slice(0, 2), geometryBoundary).toEqual([1, 0]);
  }
});

it('GPUTrajectoryRestriction decides the polygon boundary exactly, not within a tolerance band (trajectories-lines-09)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const testCase of CASES) {
    // The fixture coordinates must be exact in float32 for the hand-read expectation to hold.
    for (const value of [...testCase.ring.flat(), ...testCase.track.flat()]) {
      expect(Math.abs(Math.fround(value) - value)).toBeLessThan(1e-8);
    }
    for (const geometryBoundary of ['closed', 'open'] as const) {
      const harness = createOptimizationHarness(device);
      const output = {
        intervals: harness.output('float32x2', 2),
        positions: harness.output('float32x4', 2),
        status: harness.output('uint32', 2),
        relationships: harness.output('uint32x4', 1)
      };
      const result = await harness.run(
        new GPUTrajectoryRestriction({
          positions: harness.input('float32x2', Float32Array.from(testCase.track.flat())),
          times: harness.input('float32', Float32Array.of(0, 10)),
          offsets: harness.input('uint32', Uint32Array.of(0, 2)),
          parameters: harness.input(
            'float32',
            Float32Array.from([0, 10, 0, 0, 0, 0, 0, ...testCase.ring.flat()])
          ),
          polygonRingVertexCounts: [testCase.ring.length],
          geometryBoundary,
          output
        })
      );
      const label = `${testCase.name}, ${geometryBoundary}`;
      const expected = testCase[geometryBoundary];
      expect(result(output.relationships).slice(0, 2), label).toEqual(expected);
      const interval = result(output.intervals).slice(0, 2);
      if (expected[0] === 1) {
        expect(interval, label).toEqual([0, 10]);
      } else {
        expect(interval[0], label).toBeNaN();
        expect(interval[1], label).toBeNaN();
      }
    }
  }
});

/** Exact sign of `cross(b - a, c - a)` for coordinates that are multiples of 1/128. */
function getExactOrientation(a: number[], b: number[], c: number[]): number {
  const scaled = (value: number) => BigInt(Math.round(value * 128));
  const determinant =
    (scaled(b[0]) - scaled(a[0])) * (scaled(c[1]) - scaled(a[1])) -
    (scaled(b[1]) - scaled(a[1])) * (scaled(c[0]) - scaled(a[0]));
  return determinant > 0n ? 1 : determinant < 0n ? -1 : 0;
}

// Fix review fr-geometry-09. Two triangles and a segment that passes a vertex of each without
// touching it: on the line of each edge next to the vertex the segment is 1e-8 and 2e-8 of the
// edge's length beyond the edge's end. Whether the segment meets an edge is decided below with
// integer arithmetic, independently of the kernel: it meets none, and both fixes are outside. A
// float32 crossing test (quotients of float32 cross products) reports a crossing here under every
// evaluation order, which the boundary rule then turned into a touch of the closed polygon.
const GRAZING_CASES = [
  {
    ring: [
      [5220.0625, 48377.90625],
      [53238.078125, 67810.421875],
      [38875.796875, 25303.109375]
    ],
    track: [
      [51891.78125, 75932.34375],
      [53849.171875, 64123.828125]
    ]
  },
  {
    ring: [
      [29049.734375, 46249.640625],
      [65846.09375, 51507.484375],
      [59101.984375, 18534.1875]
    ],
    track: [
      [71396.625, 29192.609375],
      [62865.609375, 63489.96875]
    ]
  }
];

async function runSingleSegment(
  ring: number[][],
  track: number[][],
  geometryBoundary: 'closed' | 'open'
): Promise<{relationships: number[]; pieces: number[][]; pieceStatus: number[]}> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const slots = 4;
  const harness = createOptimizationHarness(device);
  const output = {
    intervals: harness.output('float32x2', 2 * slots),
    positions: harness.output('float32x4', 2 * slots),
    status: harness.output('uint32', 2 * slots),
    relationships: harness.output('uint32x4', 1)
  };
  const result = await harness.run(
    new GPUTrajectoryRestriction({
      positions: harness.input('float32x2', Float32Array.from(track.flat())),
      times: harness.input('float32', Float32Array.of(0, 10)),
      offsets: harness.input('uint32', Uint32Array.of(0, 2)),
      parameters: harness.input(
        'float32',
        Float32Array.from([0, 10, 0, 0, 0, 0, 0, ...ring.flat()])
      ),
      polygonRingVertexCounts: [ring.length],
      intervalsPerSegment: slots,
      geometryBoundary,
      output
    })
  );
  const intervals = result(output.intervals);
  const status = result(output.status);
  const pieces: number[][] = [];
  const pieceStatus: number[] = [];
  for (let slot = 0; slot < slots; slot++) {
    if ((status[slot] & 3) !== 0) {
      pieces.push([intervals[2 * slot], intervals[2 * slot + 1]]);
      pieceStatus.push(status[slot] & 3);
    }
  }
  return {relationships: result(output.relationships).slice(0, 2), pieces, pieceStatus};
}

it('GPUTrajectoryRestriction reports no crossing for a segment that passes a vertex without touching it (fr-geometry-09)', async () => {
  for (const [caseIndex, {ring, track}] of GRAZING_CASES.entries()) {
    for (const value of [...ring.flat(), ...track.flat()]) {
      expect(Math.fround(value)).toBe(value);
    }
    // Independent check of the fixture: the segment is strictly on one side of every edge it
    // could reach, or the edge is strictly on one side of the segment.
    const [first, second] = track;
    for (let vertex = 0; vertex < ring.length; vertex++) {
      const start = ring[vertex];
      const end = ring[(vertex + 1) % ring.length];
      const segmentSides =
        getExactOrientation(start, end, first) * getExactOrientation(start, end, second);
      const edgeSides =
        getExactOrientation(first, second, start) * getExactOrientation(first, second, end);
      expect(segmentSides > 0 || edgeSides > 0, `case ${caseIndex} edge ${vertex}`).toBe(true);
    }
    for (const geometryBoundary of ['closed', 'open'] as const) {
      const result = await runSingleSegment(ring, track, geometryBoundary);
      const label = `case ${caseIndex}, ${geometryBoundary}`;
      expect(result.relationships, label).toEqual([0, 0]);
      expect(result.pieces, label).toEqual([]);
    }
  }
});

it('GPUTrajectoryRestriction reports a touch for a segment through a vertex and nothing else', async () => {
  // The first triangle again. The segment runs through its second vertex exactly (the vertex is
  // the midpoint of the two fixes) in the direction (1, -1), which points away from both edges
  // at that vertex, so the vertex is the only point the segment shares with the triangle: one
  // instant at time 5 when the boundary is closed, nothing when it is open.
  const {ring} = GRAZING_CASES[0];
  const vertex = ring[1];
  const track = [
    [vertex[0] - 1024, vertex[1] + 1024],
    [vertex[0] + 1024, vertex[1] - 1024]
  ];
  expect(getExactOrientation(track[0], track[1], vertex)).toBe(0);
  expect(
    getExactOrientation(track[0], track[1], ring[0]) *
      getExactOrientation(track[0], track[1], ring[2])
  ).toBe(1);
  const closed = await runSingleSegment(ring, track, 'closed');
  expect(closed.relationships).toEqual([1, 0]);
  expect(closed.pieces).toEqual([[5, 5]]);
  expect(closed.pieceStatus).toEqual([2]);
  const open = await runSingleSegment(ring, track, 'open');
  expect(open.relationships).toEqual([0, 0]);
  expect(open.pieces).toEqual([]);
});
