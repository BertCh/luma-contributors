// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it, expect} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUFuzzyContiguity} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity';

type Point = [number, number];
/** One feature: polygons of rings of points; an empty array is an empty feature. */
type Feature = Point[][][];

const TOLERANCE = 0.1;

function square(x: number, y: number, size: number): Point[] {
  return [
    [x, y],
    [x + size, y],
    [x + size, y + size],
    [x, y + size]
  ];
}

/**
 * Builds more features than the former 512-feature cap: a 24 x 24 grid of unit squares whose
 * row and column gaps cycle through touching, within-tolerance and beyond-tolerance values, three
 * empty features, a framed hole holding one square near its edge and one far from it, and a
 * square inside the frame's solid part. Feature order is shuffled so the BVH sees unsorted input.
 */
function createFeatures(): Feature[] {
  const gaps = [0, 0.05, 0.08, 0.15, 0.3, 0.02];
  const features: Feature[] = [];
  let y = 0;
  for (let row = 0; row < 24; row++) {
    let x = 0;
    for (let column = 0; column < 24; column++) {
      features.push([[square(x, y, 1)]]);
      x += 1 + gaps[column % gaps.length];
    }
    y += 1 + gaps[(row * 5) % gaps.length];
  }
  features.push([], [], []);
  const frame: Feature = [[square(100, 100, 10), square(102, 102, 6).reverse()]];
  features.push(frame);
  features.push([[square(102.05, 104, 1)]]);
  features.push([[square(105, 105, 1)]]);
  features.push([[square(100.5, 100.5, 0.5)]]);
  let state = 12345;
  for (let index = features.length - 1; index > 0; index--) {
    state = (state * 1103515245 + 12345) % 2147483648;
    const other = state % (index + 1);
    [features[index], features[other]] = [features[other], features[index]];
  }
  return features;
}

function getPointSegmentDistance(point: Point, first: Point, second: Point): number {
  const dx = second[0] - first[0];
  const dy = second[1] - first[1];
  const square = dx * dx + dy * dy;
  const fraction =
    square > 0
      ? Math.min(Math.max(((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / square, 0), 1)
      : 0;
  return Math.hypot(point[0] - first[0] - fraction * dx, point[1] - first[1] - fraction * dy);
}

function getSegmentDistance(first: Point, second: Point, third: Point, fourth: Point): number {
  const cross = (a: Point, b: Point) => a[0] * b[1] - a[1] * b[0];
  const direction: Point = [second[0] - first[0], second[1] - first[1]];
  const other: Point = [fourth[0] - third[0], fourth[1] - third[1]];
  const offset: Point = [third[0] - first[0], third[1] - first[1]];
  const determinant = cross(direction, other);
  if (determinant !== 0) {
    const firstFraction = cross(offset, other) / determinant;
    const secondFraction = cross(offset, direction) / determinant;
    if (firstFraction >= 0 && firstFraction <= 1 && secondFraction >= 0 && secondFraction <= 1) {
      return 0;
    }
  }
  return Math.min(
    getPointSegmentDistance(first, third, fourth),
    getPointSegmentDistance(second, third, fourth),
    getPointSegmentDistance(third, first, second),
    getPointSegmentDistance(fourth, first, second)
  );
}

function containsPoint(feature: Feature, point: Point): boolean {
  return feature.some(polygon => {
    let inside = false;
    for (const ring of polygon) {
      ring.forEach((first, vertex) => {
        const second = ring[(vertex + 1) % ring.length];
        if (first[1] > point[1] !== second[1] > point[1]) {
          const intersection =
            first[0] + ((point[1] - first[1]) * (second[0] - first[0])) / (second[1] - first[1]);
          if (point[0] < intersection) inside = !inside;
        }
      });
    }
    return inside;
  });
}

function getSegments(feature: Feature): [Point, Point][] {
  return feature.flatMap(polygon =>
    polygon.flatMap(ring =>
      ring.map((point, vertex) => [point, ring[(vertex + 1) % ring.length]] as [Point, Point])
    )
  );
}

/** Brute-force CPU reference over every feature pair. */
function getReferenceDistance(first: Feature, second: Feature): number {
  const firstSegments = getSegments(first);
  const secondSegments = getSegments(second);
  if (
    firstSegments.some(([point]) => containsPoint(second, point)) ||
    secondSegments.some(([point]) => containsPoint(first, point))
  ) {
    return 0;
  }
  let best = Infinity;
  for (const [a, b] of firstSegments)
    for (const [c, d] of secondSegments) best = Math.min(best, getSegmentDistance(a, b, c, d));
  return best;
}

type FuzzyContiguityResult = {
  featureStatus: number[];
  requiredCount: number;
  overflow: number;
  offsets: number[];
  /** Neighbor rows per feature, read from the CSR. */
  neighborRows: number[][];
  added: number[];
  weights: number[];
  distances: number[];
};

/** Runs `GPUFuzzyContiguity` once with `capacity` neighbor slots and reads every output back. */
async function runFuzzyContiguity(
  features: Feature[],
  tolerance: number,
  capacity: number
): Promise<FuzzyContiguityResult> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    const rows = features.length;
    const positions: number[] = [];
    const featureOffsets = [0];
    const polygonOffsets = [0];
    const ringOffsets = [0];
    for (const feature of features) {
      for (const polygon of feature) {
        for (const ring of polygon) {
          for (const point of ring) positions.push(...point);
          ringOffsets.push(positions.length / 2);
        }
        polygonOffsets.push(ringOffsets.length - 1);
      }
      featureOffsets.push(polygonOffsets.length - 1);
    }
    let serial = 0;
    const input = <Format extends 'float32x2' | 'uint32'>(format: Format, data: number[]) => {
      const buffer = createInputBuffer(
        device,
        format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
      );
      buffers.push(buffer);
      const length = format === 'float32x2' ? data.length / 2 : data.length;
      return importGraphBuffer(graph, `input-${serial++}`, buffer, format, length);
    };
    const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
    };
    const offsets = output('uint32', rows + 1);
    const neighbors = output('uint32', capacity);
    const weights = output('float32', capacity);
    const distances = output('float32', capacity);
    const added = output('uint32', capacity);
    const featureStatus = output('uint32', rows);
    const overflow = output('uint32', 1);
    const requiredCount = output('uint32', 1);
    graph.add(
      new GPUFuzzyContiguity({
        polygons: {
          kind: 'polygons',
          positions: input('float32x2', positions),
          featureOffsets: input('uint32', featureOffsets),
          polygonOffsets: input('uint32', polygonOffsets),
          ringOffsets: input('uint32', ringOffsets)
        },
        tolerance,
        output: {
          offsets: offsets.view,
          neighbors: neighbors.view,
          weights: weights.view,
          distances: distances.view,
          added: added.view,
          featureStatus: featureStatus.view,
          status: {overflow: overflow.view, requiredCount: requiredCount.view}
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const offsetValues = await readUint32(offsets.buffer, rows + 1);
    const used = offsetValues[rows];
    const neighborValues = await readUint32(neighbors.buffer, used);
    return {
      featureStatus: await readUint32(featureStatus.buffer, rows),
      requiredCount: (await readUint32(requiredCount.buffer, 1))[0],
      overflow: (await readUint32(overflow.buffer, 1))[0],
      offsets: offsetValues,
      neighborRows: features.map((_, row) =>
        neighborValues.slice(offsetValues[row], offsetValues[row + 1])
      ),
      added: await readUint32(added.buffer, used),
      weights: await readFloat32(weights.buffer, used),
      distances: await readFloat32(distances.buffer, used)
    };
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

/** Every listed pair appears in both rows. */
function expectSymmetric(neighborRows: number[][]): void {
  const asymmetric: [number, number][] = [];
  neighborRows.forEach((neighbors, row) => {
    for (const other of neighbors)
      if (!neighborRows[other].includes(row)) asymmetric.push([row, other]);
  });
  expect(asymmetric).toEqual([]);
}

/** Brute-force CPU neighbor rows over every feature pair. */
function getReferenceNeighbors(features: Feature[], tolerance: number): number[][] {
  return features.map((first, row) =>
    first.length === 0
      ? []
      : features.flatMap((second, other) =>
          other !== row && second.length > 0 && getReferenceDistance(first, second) <= tolerance
            ? [other]
            : []
        )
  );
}

/** Shuffles in place with a fixed linear congruential sequence. */
function shuffle<Item>(items: Item[], seed: number): Item[] {
  let state = seed;
  for (let index = items.length - 1; index > 0; index--) {
    state = (state * 1103515245 + 12345) % 2147483648;
    const other = state % (index + 1);
    [items[index], items[other]] = [items[other], items[index]];
  }
  return items;
}

/** The next float32 above a float32 value. */
function getNextFloat32(value: number): number {
  const float = new Float32Array([value]);
  const bits = new Uint32Array(float.buffer);
  bits[0] += value >= 0 ? 1 : -1;
  return float[0];
}

function rectangle(
  minimumX: number,
  minimumY: number,
  maximumX: number,
  maximumY: number
): Point[] {
  return [
    [minimumX, minimumY],
    [maximumX, minimumY],
    [maximumX, maximumY],
    [minimumX, maximumY]
  ];
}

it('fuzzy contiguity lists a pair at the tolerance from both sides when the tolerance dwarfs the coordinates', async () => {
  // Review input: the float32 edge gap is exactly 100, but A's query box, widened by a slack that
  // scaled only with A's small coordinates, rounded to just below B's minimum x.
  const tolerance = 100;
  const first: Feature = [[rectangle(0.0861, 0, 0.1861, 0.1)]];
  const second: Feature = [[rectangle(100.18610382080078, 0, 100.3, 0.1)]];
  const result = await runFuzzyContiguity([first, second], tolerance, 2);
  expect(result.featureStatus).toEqual([0, 0]);
  expect(result.neighborRows).toEqual([[1], [0]]);
  expect(result.requiredCount).toBe(2);
  expect(result.overflow).toBe(0);
});

it('fuzzy contiguity stays symmetric on shuffled pairs one float32 step around the tolerance', async () => {
  // For each small square A, square B starts one float32 step above A's maximum x plus the
  // tolerance, and square C two steps further: gaps within rounding of the tolerance. The first
  // four maxima are the review's search hits, where the float32 gap B - A is exactly 100.
  const tolerance = 100;
  const features: Feature[] = [];
  const maxima = [1861, 2486, 3111, 3736, 730, 1460, 2190, 2920].map(step =>
    Math.fround(step * 1e-4)
  );
  for (const [index, maximumX] of maxima.entries()) {
    const edge = Math.fround(maximumX + Math.fround(tolerance + Math.fround(maximumX * 1e-5)));
    const nearMinimumX = getNextFloat32(edge);
    const farMinimumX = getNextFloat32(getNextFloat32(nearMinimumX));
    const y = index * 1000;
    features.push([[rectangle(maximumX - 0.1, y, maximumX, y + 0.1)]]);
    features.push([[rectangle(nearMinimumX, y, nearMinimumX + 0.2, y + 0.1)]]);
    features.push([[rectangle(farMinimumX, y + 0.5, farMinimumX + 0.2, y + 0.6)]]);
  }
  shuffle(features, 777);
  const rows = features.length;
  const result = await runFuzzyContiguity(features, tolerance, rows * rows);
  expect(result.overflow).toBe(0);
  expectSymmetric(result.neighborRows);
  // Every pair the double-precision reference accepts is found; any extra pair is within float32
  // rounding of the tolerance.
  const reference = getReferenceNeighbors(features, tolerance);
  result.neighborRows.forEach((neighbors, row) => {
    for (const other of reference[row]) expect(neighbors).toContain(other);
    for (const other of neighbors)
      expect(getReferenceDistance(features[row], features[other])).toBeLessThanOrEqual(
        tolerance * (1 + 1e-6)
      );
  });
});

it('fuzzy contiguity accepts a gap equal to the tolerance and rejects the next larger gap', async () => {
  // Dyadic coordinates: float32 and double precision agree exactly.
  const tolerance = 0.25;
  const features: Feature[] = shuffle(
    [
      [[square(0, 0, 1)]],
      [[square(1.25, 0, 1)]],
      [[square(2.5, 0, 1)]],
      [[square(3.75 + 0.0078125, 0, 1)]],
      [[square(0, 1.25, 1)]],
      [[square(0, 2.5 + 0.0078125, 1)]]
    ],
    99
  );
  const result = await runFuzzyContiguity(features, tolerance, 36);
  expect(result.neighborRows).toEqual(getReferenceNeighbors(features, tolerance));
  expect(result.distances.every(value => value === tolerance)).toBe(true);
  expect(result.added.every(value => value === 1)).toBe(true);
});

it('fuzzy contiguity handles a tolerance far larger than the extent and gaps at a large tolerance', async () => {
  // The small squares are neighbors of each other; the square at x = 2^20 + 1 sits exactly 2^20
  // from the first square, within 2^20 of the second and beyond it from the third; the square
  // near x = -2^21 is isolated. Under a tolerance of 1e17 every pair is a neighbor.
  const tolerance = 2 ** 20;
  const features: Feature[] = shuffle(
    [
      [[square(0, 0, 1)]],
      [[square(3, 2, 1)]],
      [[square(-5, 4, 2)]],
      [[rectangle(1 + tolerance, 0, 2 + tolerance, 1)]],
      [[rectangle(-(2 ** 21) - 5, 4, -(2 ** 21) - 4, 5)]]
    ],
    5
  );
  const result = await runFuzzyContiguity(features, tolerance, 25);
  expect(result.neighborRows).toEqual(getReferenceNeighbors(features, tolerance));
  expectSymmetric(result.neighborRows);
  const huge = await runFuzzyContiguity(features, 1e17, 25);
  expect(huge.neighborRows).toEqual(
    features.map((_, row) => features.flatMap((__, other) => (other === row ? [] : [other])))
  );
});

it('fuzzy contiguity is exact for coordinates of large magnitude', async () => {
  // Near 2^22 the float32 step is 0.5: gaps cycle through 0 (touching), 0.5 (accepted at
  // tolerance 0.5) and 1 (rejected), along both axes.
  const origin = 2 ** 22;
  const tolerance = 0.5;
  const gaps = [0, 0.5, 1];
  const starts = [0];
  for (let index = 1; index < 6; index++)
    starts.push(starts[index - 1] + 7.5 + gaps[(index - 1) % gaps.length]);
  const features: Feature[] = [];
  for (let row = 0; row < 6; row++)
    for (let column = 0; column < 6; column++) {
      const x = origin + starts[column];
      const y = -origin - 64 + starts[row];
      features.push([[rectangle(x, y, x + 7.5, y + 7.5)]]);
    }
  shuffle(features, 31);
  const rows = features.length;
  const result = await runFuzzyContiguity(features, tolerance, rows * rows);
  expect(result.neighborRows).toEqual(getReferenceNeighbors(features, tolerance));
  expectSymmetric(result.neighborRows);
  expect(result.added).toContain(1);
});

it('indexed fuzzy contiguity matches a brute-force reference above the former 512-feature cap', async () => {
  const features = createFeatures();
  const rows = features.length;
  expect(rows).toBeGreaterThan(512);
  const expectedNeighbors = getReferenceNeighbors(features, TOLERANCE);
  const expectedAdded = expectedNeighbors.map((neighbors, row) =>
    neighbors.map(other => (getReferenceDistance(features[row], features[other]) > 0 ? 1 : 0))
  );
  const required = expectedNeighbors.reduce((sum, neighbors) => sum + neighbors.length, 0);
  const result = await runFuzzyContiguity(features, TOLERANCE, required);
  expect(result.featureStatus).toEqual(features.map(feature => (feature.length === 0 ? 1 : 0)));
  expect(result.requiredCount).toBe(required);
  expect(result.overflow).toBe(0);
  const expectedOffsets = [0];
  for (const row of expectedNeighbors)
    expectedOffsets.push(expectedOffsets[expectedOffsets.length - 1] + row.length);
  expect(result.offsets).toEqual(expectedOffsets);
  expect(result.neighborRows).toEqual(expectedNeighbors);
  expectSymmetric(result.neighborRows);
  expect(result.added).toEqual(expectedAdded.flat());
  expect(result.weights).toEqual(new Array(required).fill(1));
  expect(result.distances.every(value => value >= 0 && value <= TOLERANCE)).toBe(true);
  // The grid contributes all three relations: touching, separated and within tolerance.
  expect(expectedAdded.flat()).toContain(0);
  expect(expectedAdded.flat()).toContain(1);
});

it('fuzzy contiguity rejects more features than its documented limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  expect(
    () =>
      new GPUFuzzyContiguity({
        polygons: {
          kind: 'polygons',
          positions: view('float32x2', 16),
          featureOffsets: view('uint32', 65538),
          polygonOffsets: view('uint32', 5),
          ringOffsets: view('uint32', 5)
        },
        tolerance: TOLERANCE,
        output: {
          offsets: view('uint32', 65538),
          neighbors: view('uint32', 8),
          weights: view('float32', 8),
          added: view('uint32', 8),
          featureStatus: view('uint32', 65537),
          status: {overflow: view('uint32', 1)}
        }
      })
  ).toThrow(/^fuzzy-contiguity features must be an integer in \[1, 65536\]$/);
});
