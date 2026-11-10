// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-search';
import {createOptimizationHarness} from '../../gpu-network/network-optimization/optimization-test-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRandom} from '../trajectory-interpolation/trajectory-interpolation-fixtures';

type Polyline = [number, number][];
type SearchMetric = ConstructorParameters<typeof GPUTrackSimilaritySearch>[0]['metric'];

const EARTH_RADIUS = 6371008.8;
const GREAT_CIRCLE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  sphereRadius: EARTH_RADIUS,
  units: 'meters'
};

function pack(polylines: readonly Polyline[]) {
  const positions: number[] = [];
  const offsets = [0];
  for (const polyline of polylines) {
    for (const [x, y] of polyline) positions.push(x, y);
    offsets.push(positions.length / 2);
  }
  return {positions: Float32Array.from(positions), offsets: Uint32Array.from(offsets)};
}

function createWalks(
  seed: number,
  count: number,
  center: [number, number],
  spread: number,
  step: number
): Polyline[] {
  const random = createRandom(seed);
  return Array.from({length: count}, () => {
    let x = center[0] + (random() - 0.5) * spread;
    let y = center[1] + (random() - 0.5) * spread;
    return Array.from({length: 6 + Math.floor(random() * 6)}, () => {
      x += (random() - 0.4) * step;
      y += (random() - 0.5) * step;
      return [Math.fround(x), Math.fround(y)] as [number, number];
    });
  });
}

async function runSearch(
  tracksA: readonly Polyline[],
  tracksB: readonly Polyline[],
  pairA: number[],
  pairB: number[],
  options: {
    metric: SearchMetric;
    threshold: number;
    topK?: number;
    spatialContext?: GPUSpatialContext;
    editDistanceTolerance?: number;
    lcssTolerance?: number;
  }
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const packedA = pack(tracksA);
  const packedB = pack(tracksB);
  const pairCount = pairA.length;
  const subsequence = options.metric === 'longest-common-subsequence';
  const output = {
    distance: subsequence ? undefined : harness.output('float32', pairCount),
    lowerBound: subsequence ? undefined : harness.output('float32', pairCount),
    similarity: subsequence ? harness.output('float32', pairCount) : undefined,
    upperBound: subsequence ? harness.output('float32', pairCount) : undefined,
    matched: harness.output('uint32', pairCount),
    status: harness.output('uint32', pairCount),
    topKIndices: options.topK ? harness.output('uint32', tracksA.length * options.topK) : undefined,
    topKSummary: options.topK ? harness.output('uint32x2', tracksA.length) : undefined
  };
  const result = await harness.run(
    new GPUTrackSimilaritySearch({
      metric: options.metric,
      spatialContext: options.spatialContext,
      editDistanceTolerance: options.editDistanceTolerance,
      lcssTolerance: options.lcssTolerance,
      topK: options.topK,
      linesA: {
        kind: 'lines',
        positions: harness.input('float32x2', packedA.positions),
        lineOffsets: harness.input('uint32', packedA.offsets)
      },
      linesB: {
        kind: 'lines',
        positions: harness.input('float32x2', packedB.positions),
        lineOffsets: harness.input('uint32', packedB.offsets)
      },
      pairA: harness.input('uint32', Uint32Array.from(pairA)),
      pairB: harness.input('uint32', Uint32Array.from(pairB)),
      threshold: harness.input('float32', Float32Array.of(options.threshold)),
      output: output as never
    })
  );
  const value = subsequence ? output.similarity! : output.distance!;
  const bound = subsequence ? output.upperBound! : output.lowerBound!;
  return {
    value: result(value),
    bound: result(bound),
    matched: result(output.matched),
    status: result(output.status),
    topKIndices: output.topKIndices ? result(output.topKIndices) : [],
    topKSummary: output.topKSummary ? result(output.topKSummary) : []
  };
}

/**
 * Selection reference that does not look at the kernel: per A track the matched rows ordered by
 * value (ties by row), and whether any of its rows has an error status.
 */
function selectReferenceTopK(
  result: {value: number[]; matched: number[]; status: number[]},
  pairA: number[],
  trackCount: number,
  topK: number,
  descending: boolean
) {
  return Array.from({length: trackCount}, (_, track) => {
    const rows = pairA.flatMap((owner, row) => (owner === track ? [row] : []));
    const matchedRows = rows
      .filter(row => result.matched[row] === 1)
      .sort(
        (first, second) =>
          (descending
            ? result.value[second] - result.value[first]
            : result.value[first] - result.value[second]) || first - second
      );
    return {
      rows: matchedRows.slice(0, topK),
      complete: rows.every(row => result.status[row] <= 1)
    };
  });
}

for (const metric of ['hausdorff', 'longest-common-subsequence'] as const) {
  it(`GPUTrackSimilaritySearch segmented top-k (${metric}) selects like a per-track stable sort`, async () => {
    const descending = metric === 'longest-common-subsequence';
    const queries = createWalks(3, 7, [0, 0], 4, 1.2);
    const candidates = createWalks(5, 24, [0, 0], 6, 1.2);
    // Duplicate candidates give exactly equal values, so ties are decided by pair row.
    candidates.push(candidates[2], candidates[2], candidates[9]);
    const random = createRandom(17);
    const pairA: number[] = [];
    const pairB: number[] = [];
    for (let query = 0; query < queries.length; query++) {
      for (let candidate = 0; candidate < candidates.length; candidate++) {
        if (random() < 0.8) {
          pairA.push(query);
          pairB.push(candidate);
        }
      }
    }
    // Error pairs: a candidate that does not exist (status 2) for tracks 1 and 4.
    pairA.push(1, 4);
    pairB.push(999, 999);
    // Shuffle so the pairs of a track are not contiguous.
    for (let row = pairA.length - 1; row > 0; row--) {
      const other = Math.floor(random() * (row + 1));
      [pairA[row], pairA[other]] = [pairA[other], pairA[row]];
      [pairB[row], pairB[other]] = [pairB[other], pairB[row]];
    }
    const topK = 5;
    const result = await runSearch(queries, candidates, pairA, pairB, {
      metric,
      topK,
      threshold: descending ? 0 : 1e9,
      lcssTolerance: 1
    });
    const reference = selectReferenceTopK(result, pairA, queries.length, topK, descending);
    // Teeth: ties and error pairs are present, and the selection is not trivially the whole list.
    const tied = reference.some(({rows}) =>
      rows.some((row, index) => index > 0 && result.value[row] === result.value[rows[index - 1]])
    );
    expect(tied, 'ties in the selected rows').toBe(true);
    expect(result.status.filter(status => status > 1).length).toBe(2);
    expect(reference.filter(({complete}) => !complete).length).toBe(2);
    expect(reference.every(({rows}) => rows.length === topK)).toBe(true);
    reference.forEach(({rows, complete}, track) => {
      const selected = result.topKIndices.slice(track * topK, (track + 1) * topK);
      expect(selected, `track ${track}`).toEqual(
        Array.from({length: topK}, (_, slot) => rows[slot] ?? 0xffffffff)
      );
      expect(result.topKSummary.slice(2 * track, 2 * track + 2), `track ${track} summary`).toEqual([
        rows.length,
        complete ? 1 : 0
      ]);
    });
  });
}

it('GPUTrackSimilaritySearch segmented top-k leaves unmatched slots empty and tracks without pairs empty', async () => {
  const queries = createWalks(2, 3, [0, 0], 2, 1);
  const candidates = createWalks(4, 4, [0, 0], 2, 1);
  // Track 1 has no pair at all, track 2 only a pair above the threshold.
  const result = await runSearch(queries, candidates, [0, 0, 2], [0, 1, 3], {
    metric: 'hausdorff',
    topK: 4,
    threshold: 1e-6
  });
  const reference = selectReferenceTopK(result, [0, 0, 2], 3, 4, false);
  expect(result.topKIndices.slice(4, 8)).toEqual(new Array(4).fill(0xffffffff));
  expect(result.topKSummary.slice(2, 4)).toEqual([0, 1]);
  reference.forEach(({rows}, track) => expect(rows.length).toBe(result.topKSummary[2 * track]));
});

// Independent binary64 great-circle references on the same f32-rounded vertices.
function haversine(a: [number, number], b: [number, number]): number {
  const toRadians = Math.PI / 180;
  const deltaLatitude = (b[1] - a[1]) * toRadians;
  const deltaLongitude = (b[0] - a[0]) * toRadians;
  const term =
    Math.sin(deltaLatitude / 2) ** 2 +
    Math.cos(a[1] * toRadians) * Math.cos(b[1] * toRadians) * Math.sin(deltaLongitude / 2) ** 2;
  return 2 * EARTH_RADIUS * Math.asin(Math.min(1, Math.sqrt(term)));
}

function referenceDynamicTimeWarping(a: Polyline, b: Polyline): number {
  const table = Array.from({length: a.length + 1}, () => new Array(b.length + 1).fill(Infinity));
  table[0][0] = 0;
  for (let row = 1; row <= a.length; row++)
    for (let column = 1; column <= b.length; column++)
      table[row][column] =
        haversine(a[row - 1], b[column - 1]) +
        Math.min(table[row - 1][column], table[row][column - 1], table[row - 1][column - 1]);
  return table[a.length][b.length];
}

function referenceFrechet(a: Polyline, b: Polyline): number {
  const table = Array.from({length: a.length}, () => new Array(b.length).fill(0));
  for (let row = 0; row < a.length; row++)
    for (let column = 0; column < b.length; column++) {
      const distance = haversine(a[row], b[column]);
      const previous =
        row === 0 && column === 0
          ? 0
          : Math.min(
              row > 0 ? table[row - 1][column] : Infinity,
              column > 0 ? table[row][column - 1] : Infinity,
              row > 0 && column > 0 ? table[row - 1][column - 1] : Infinity
            );
      table[row][column] = Math.max(distance, previous);
    }
  return table[a.length - 1][b.length - 1];
}

function referenceEditDistanceWithRealPenalty(a: Polyline, b: Polyline): number {
  const gap: [number, number] = [0, 0];
  const table = Array.from({length: a.length + 1}, () => new Array(b.length + 1).fill(0));
  for (let row = 1; row <= a.length; row++)
    table[row][0] = table[row - 1][0] + haversine(a[row - 1], gap);
  for (let column = 1; column <= b.length; column++)
    table[0][column] = table[0][column - 1] + haversine(b[column - 1], gap);
  for (let row = 1; row <= a.length; row++)
    for (let column = 1; column <= b.length; column++)
      table[row][column] = Math.min(
        table[row - 1][column - 1] + haversine(a[row - 1], b[column - 1]),
        table[row - 1][column] + haversine(a[row - 1], gap),
        table[row][column - 1] + haversine(b[column - 1], gap)
      );
  return table[a.length][b.length];
}

function referenceEditDistanceOnRealSequence(a: Polyline, b: Polyline, tolerance: number): number {
  const table = Array.from({length: a.length + 1}, (_, row) =>
    Array.from({length: b.length + 1}, (_, column) => (row === 0 ? column : column === 0 ? row : 0))
  );
  for (let row = 1; row <= a.length; row++)
    for (let column = 1; column <= b.length; column++)
      table[row][column] = Math.min(
        table[row - 1][column - 1] + (haversine(a[row - 1], b[column - 1]) <= tolerance ? 0 : 1),
        table[row - 1][column] + 1,
        table[row][column - 1] + 1
      );
  return table[a.length][b.length];
}

const GREAT_CIRCLE_MEASURES: {
  metric: SearchMetric;
  reference: (a: Polyline, b: Polyline) => number;
  tolerance?: number;
}[] = [
  {metric: 'dynamic-time-warping', reference: referenceDynamicTimeWarping},
  {metric: 'frechet', reference: referenceFrechet},
  {metric: 'edit-distance-with-real-penalty', reference: referenceEditDistanceWithRealPenalty},
  {
    metric: 'edit-distance-on-real-sequence',
    reference: (a, b) => referenceEditDistanceOnRealSequence(a, b, 4000),
    tolerance: 4000
  }
];

for (const {metric, reference, tolerance} of GREAT_CIRCLE_MEASURES) {
  it(`GPUTrackSimilaritySearch ${metric} measures great-circle distance and prunes with admissible bounds`, async () => {
    // High latitude, where a degree of longitude is much shorter than a degree of latitude.
    const queries = createWalks(21, 3, [10, 66], 0.2, 0.05);
    const candidates = [
      ...createWalks(23, 14, [10, 66], 0.8, 0.05),
      // Far candidates, whose endpoint distances alone exceed the threshold.
      ...createWalks(29, 6, [14, 67.5], 0.3, 0.05)
    ];
    const pairA: number[] = [];
    const pairB: number[] = [];
    for (let query = 0; query < queries.length; query++)
      for (let candidate = 0; candidate < candidates.length; candidate++) {
        pairA.push(query);
        pairB.push(candidate);
      }
    const expected = pairA.map((query, row) => reference(queries[query], candidates[pairB[row]]));
    const options = {
      metric,
      spatialContext: GREAT_CIRCLE,
      editDistanceTolerance: tolerance
    };
    const unpruned = await runSearch(queries, candidates, pairA, pairB, {
      ...options,
      threshold: 1e12
    });
    expect(unpruned.status.every(status => status === 0)).toBe(true);
    expected.forEach((value, row) => {
      expect(Math.abs(unpruned.value[row] - value), `${metric} value ${row}`).toBeLessThanOrEqual(
        1e-4 * Math.max(value, 1) + 1
      );
      // The bound never exceeds the measure in the same arithmetic.
      expect(unpruned.bound[row], `${metric} bound ${row}`).toBeLessThanOrEqual(
        unpruned.value[row] * (1 + 1e-5) + 1e-3
      );
    });
    // A median threshold prunes some pairs and keeps exactly the matches of the unpruned values.
    const threshold = [...expected].sort((first, second) => first - second)[
      Math.floor(expected.length * 0.5)
    ];
    const pruned = await runSearch(queries, candidates, pairA, pairB, {...options, threshold});
    const prunedCount = pruned.status.filter(status => status === 1).length;
    if (metric === 'dynamic-time-warping' || metric === 'frechet') {
      expect(prunedCount, `${metric} pruned pairs`).toBeGreaterThan(0);
    }
    pruned.status.forEach((status, row) => {
      if (status === 1)
        expect(expected[row], `${metric} pruned row ${row}`).toBeGreaterThan(threshold);
    });
    expected.forEach((value, row) => {
      // Skip values within f32 rounding of the threshold.
      if (Math.abs(value - threshold) <= 1e-4 * Math.max(threshold, 1) + 1) return;
      expect(pruned.matched[row], `${metric} matched ${row}`).toBe(value <= threshold ? 1 : 0);
    });
  });
}

it('GPUTrackSimilaritySearch planar and great-circle contexts disagree on longitude-latitude input', async () => {
  const queries = createWalks(21, 2, [10, 66], 0.2, 0.05);
  const candidates = createWalks(23, 3, [10, 66], 0.8, 0.05);
  const pairA = [0, 0, 0, 1, 1, 1];
  const pairB = [0, 1, 2, 0, 1, 2];
  const planar = await runSearch(queries, candidates, pairA, pairB, {
    metric: 'dynamic-time-warping',
    threshold: 1e12
  });
  const greatCircle = await runSearch(queries, candidates, pairA, pairB, {
    metric: 'dynamic-time-warping',
    threshold: 1e12,
    spatialContext: GREAT_CIRCLE
  });
  // Planar values are degrees (below 100), great-circle values are meters (thousands).
  expect(Math.max(...planar.value)).toBeLessThan(100);
  expect(Math.min(...greatCircle.value)).toBeGreaterThan(1000);
});

it('GPUTrackSimilaritySearch rejects unsupported spatial contexts', () => {
  expect(
    () =>
      new GPUTrackSimilaritySearch({
        metric: 'hausdorff',
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'native'} as never
      } as never)
  ).toThrow(/incompatible/);
});
