// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRandom} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {runSearch, type SearchMetric} from './track-similarity-edit-measures-harness';
import {
  computeEditDistanceOnRealSequenceLowerBoundOracle,
  computeEditDistanceOnRealSequenceOracle,
  computeEditDistanceWithRealPenaltyLowerBoundOracle,
  computeEditDistanceWithRealPenaltyOracle,
  computeHausdorffLowerBoundOracle,
  computeSymmetricSegmentPathLowerBoundOracle,
  computeSymmetricSegmentPathOracle
} from './track-similarity-edit-oracle';
import {computeHausdorffOracle, type Polyline} from './track-similarity-oracle';

const TOLERANCE = 2;
const GAP: [number, number] = [0, 0];
const QUERY_COUNT = 4;
const CANDIDATE_COUNT = 30;
const TOP_K = 3;

const MEASURES: Record<
  string,
  {
    metric: SearchMetric;
    value: (a: Polyline, b: Polyline) => number;
    bound: (a: Polyline, b: Polyline) => number;
  }
> = {
  hausdorff: {
    metric: 'hausdorff',
    value: (a, b) => computeHausdorffOracle(a, b),
    bound: computeHausdorffLowerBoundOracle
  },
  sspd: {
    metric: 'symmetric-segment-path',
    value: computeSymmetricSegmentPathOracle,
    bound: computeSymmetricSegmentPathLowerBoundOracle
  },
  edr: {
    metric: 'edit-distance-on-real-sequence',
    value: (a, b) => computeEditDistanceOnRealSequenceOracle(a, b, TOLERANCE),
    bound: (a, b) => computeEditDistanceOnRealSequenceLowerBoundOracle(a, b, TOLERANCE)
  },
  erp: {
    metric: 'edit-distance-with-real-penalty',
    value: (a, b) => computeEditDistanceWithRealPenaltyOracle(a, b, GAP),
    bound: (a, b) => computeEditDistanceWithRealPenaltyLowerBoundOracle(a, b, GAP)
  }
};

/** Query walks near the origin and candidates spread along x, so most candidates are far. */
function createScene() {
  const random = createRandom(31);
  const fround = (value: number) => Math.fround(value);
  const walk = (count: number, originX: number, originY: number): [number, number][] => {
    let [x, y] = [originX, originY];
    return Array.from({length: count}, () => {
      x += (random() - 0.3) * 2;
      y += (random() - 0.5) * 2;
      return [fround(x), fround(y)];
    });
  };
  const queries = Array.from({length: QUERY_COUNT}, (_, index) =>
    walk(15 + index * 5, random() * 4, random() * 4)
  );
  const candidates = Array.from({length: CANDIDATE_COUNT}, (_, index) =>
    walk(10 + Math.floor(random() * 30), index * 12 + random() * 5, (index % 5) * 6)
  );
  const pairA = new Uint32Array(QUERY_COUNT * CANDIDATE_COUNT);
  const pairB = new Uint32Array(QUERY_COUNT * CANDIDATE_COUNT);
  for (let query = 0; query < QUERY_COUNT; query++) {
    for (let candidate = 0; candidate < CANDIDATE_COUNT; candidate++) {
      pairA[query * CANDIDATE_COUNT + candidate] = query;
      pairB[query * CANDIDATE_COUNT + candidate] = candidate;
    }
  }
  return {queries, candidates, pairA, pairB};
}

/** The CPU top-k rows per query: smallest value first, ties by lower row. */
function getReferenceTopK(values: readonly number[], pairA: Uint32Array): number[][] {
  return Array.from({length: QUERY_COUNT}, (_, query) =>
    values
      .map((value, row) => ({value, row}))
      .filter(({row}) => pairA[row] === query)
      .sort((first, second) => first.value - second.value || first.row - second.row)
      .slice(0, TOP_K)
      .map(({row}) => row)
  );
}

function readTopK(indices: Uint32Array): number[][] {
  return Array.from({length: QUERY_COUNT}, (_, query) =>
    Array.from(indices.slice(query * TOP_K, (query + 1) * TOP_K)).filter(row => row !== 0xffffffff)
  );
}

for (const [name, measure] of Object.entries(MEASURES)) {
  it(`GPUTrackSimilaritySearch ${name}: bounds never exceed the distance and pruned top-k equals unpruned top-k`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {queries, candidates, pairA, pairB} = createScene();
    const options = {
      metric: measure.metric,
      topK: TOP_K,
      editDistanceTolerance: TOLERANCE,
      editDistanceGapPoint: GAP
    };
    const reference = Array.from(pairA, (query, row) =>
      measure.value(queries[query], candidates[pairB[row]])
    );
    const referenceBound = Array.from(pairA, (query, row) =>
      measure.bound(queries[query], candidates[pairB[row]])
    );
    // Double-precision bounds are below the double-precision distances (no tolerance).
    referenceBound.forEach((bound, row) =>
      expect(bound, `${name} row ${row}`).toBeLessThanOrEqual(reference[row])
    );

    // Unpruned: a threshold above every distance evaluates every pair.
    const unpruned = await runSearch(device, queries, candidates, pairA, pairB, {
      ...options,
      threshold: 1e9
    });
    expect(
      Array.from(unpruned.status).every(status => status === 0),
      'unpruned statuses'
    ).toBe(true);
    reference.forEach((value, row) => {
      expect(unpruned.distance[row], `${name} distance ${row}`).toBeCloseTo(
        value,
        3 - Math.floor(Math.log10(Math.max(1, value))) + 1
      );
      // The GPU bound is below the GPU distance, up to f32 rounding of the distance itself.
      expect(unpruned.lowerBound[row], `${name} bound ${row}`).toBeLessThanOrEqual(
        unpruned.distance[row] * (1 + 1e-5) + 1e-4
      );
      // And it is the bound of the oracle (the kernel subtracts a rounding slack that grows with
      // the number of summed terms): bound kernels are not trivially zero.
      expect(
        Math.abs(unpruned.lowerBound[row] - referenceBound[row]),
        `${name} bound vs oracle ${row}`
      ).toBeLessThanOrEqual(1e-4 * referenceBound[row] + 1e-3);
    });
    const referenceTopK = getReferenceTopK(reference, pairA);
    expect(readTopK(unpruned.topKIndices), `${name} unpruned top-k`).toEqual(referenceTopK);

    // Pruned: the threshold is the largest k-th smallest distance over the queries (the smallest
    // threshold that keeps every top-k pair), widened by 1e-4 for the f32 rounding of the
    // distances, so far candidates are pruned by their bounds.
    const kthLargest =
      Math.max(...referenceTopK.map(rows => reference[rows[TOP_K - 1]])) * (1 + 1e-4) + 1e-6;
    const pruned = await runSearch(device, queries, candidates, pairA, pairB, {
      ...options,
      threshold: kthLargest
    });
    const prunedCount = Array.from(pruned.status).filter(status => status === 1).length;
    // Teeth: pruning happens, and only pairs whose true distance exceeds the threshold are pruned.
    expect(prunedCount, `${name} pruned pairs`).toBeGreaterThan(2 * QUERY_COUNT);
    pruned.status.forEach((status, row) => {
      if (status === 1) {
        expect(reference[row], `${name} pruned row ${row}`).toBeGreaterThan(kthLargest);
        expect(Number.isNaN(pruned.distance[row])).toBe(true);
      }
    });
    expect(readTopK(pruned.topKIndices), `${name} pruned top-k`).toEqual(referenceTopK);
    expect(readTopK(pruned.topKIndices)).toEqual(readTopK(unpruned.topKIndices));
    expect(Array.from(pruned.topKSummary).filter((_, slot) => slot % 2 === 1)).toEqual([
      1, 1, 1, 1
    ]);
  }, 120000);
}
