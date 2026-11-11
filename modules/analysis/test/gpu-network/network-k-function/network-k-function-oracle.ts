// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PhiloxStream} from '../../../src/analysis-core/kernels/permutation-random';
import {
  getNetworkKFunctionChunkCount,
  getNetworkKFunctionLengthRange,
  NETWORK_K_FUNCTION_CHUNK_EDGE_COUNT,
  NETWORK_K_FUNCTION_EDGE_LENGTH_RANGE,
  NETWORK_K_FUNCTION_STREAM
} from '../../../src/gpu-network/network-k-function/gpu-network-k-function';
import {
  dijkstra,
  NONE,
  snapOracle,
  type AccessibilityCSR
} from '../network-accessibility/network-accessibility-oracle';

/** One event placed on the network: edge endpoints and the costs to them, or inactive. */
export type KFunctionEvent = {
  first: number;
  second: number;
  firstCost: number;
  secondCost: number;
};

const INACTIVE: KFunctionEvent = {first: NONE, second: NONE, firstCost: -1, secondCost: -1};

/** Snaps events like `GPUNetworkSnapping` with `startDirection: 'both'`. */
export function snapEvents(
  points: ArrayLike<number>,
  positions: ArrayLike<number>,
  csr: AccessibilityCSR,
  maximumSnapDistance: number = Infinity
): KFunctionEvent[] {
  return snapOracle(points, positions, csr.sources, csr.neighbors, {
    edgeCosts: csr.weights,
    maximumSnapDistance
  }).map(snap =>
    snap.edge === NONE
      ? INACTIVE
      : {
          first: csr.sources[snap.edge],
          second: csr.neighbors[snap.edge],
          firstCost: Math.fround(snap.sourceCost),
          secondCost: Math.fround(snap.targetCost)
        }
  );
}

/**
 * Draws `eventCount` random events like the GPU `simulate` kernel: a chunk of 128 slots and then
 * an edge of it, each with probability proportional to its quantized length, through a Philox
 * stream keyed by `seed`, `(event, pattern)` counters and the K function tag, then a uniform
 * fraction along the edge. This is a mirror of the kernel's integers, so it pins the schedule and
 * the stream, not the proportions; the spec "draws short edges in proportion to their length"
 * checks those against the exact length shares.
 */
export function simulateEvents(
  csr: AccessibilityCSR,
  nodeCount: number,
  seed: number,
  pattern: number,
  eventCount: number,
  validCount: number
): KFunctionEvent[] {
  // The GPU quantizes against the longest live edge, not networkLength (review B-24).
  const isLive = (edge: number) =>
    edge < csr.offsets[nodeCount] &&
    Number.isFinite(csr.weights[edge]) &&
    csr.weights[edge] >= 0 &&
    csr.neighbors[edge] < nodeCount;
  let maximum = 0;
  for (let edge = 0; edge < csr.neighbors.length; edge++) {
    if (isLive(edge)) {
      maximum = Math.max(maximum, csr.weights[edge]);
    }
  }
  // Two-level draw (review network-paths-04): edge lengths in 2^24 units of the longest live
  // edge, a prefix sum inside each chunk of 128 slots, and a prefix sum over the chunk totals
  // shifted right until the largest is below the coarse range. Both roundings are to nearest.
  const edgeCount = csr.neighbors.length;
  const chunkCount = getNetworkKFunctionChunkCount(edgeCount);
  const prefix = new Uint32Array(edgeCount);
  const chunkTotals = new Uint32Array(chunkCount);
  for (let edge = 0; edge < edgeCount; edge++) {
    const chunk = Math.floor(edge / NETWORK_K_FUNCTION_CHUNK_EDGE_COUNT);
    if (maximum > 0 && isLive(edge)) {
      const scaled =
        Math.min(Math.fround(csr.weights[edge] / maximum), 1) *
        NETWORK_K_FUNCTION_EDGE_LENGTH_RANGE;
      const whole = Math.floor(scaled);
      chunkTotals[chunk] += whole + (scaled - whole >= 0.5 ? 1 : 0);
    }
    prefix[edge] = chunkTotals[chunk];
  }
  const chunkMaximum = chunkTotals.reduce((left, right) => Math.max(left, right), 0);
  const maximumBits = 32 - Math.clz32(chunkMaximum);
  const chunkRangeBits = Math.log2(getNetworkKFunctionLengthRange(chunkCount));
  const shift = Math.max(maximumBits - chunkRangeBits, 0);
  const chunkPrefix = new Float64Array(chunkCount);
  let sum = 0;
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    const total = chunkTotals[chunk];
    sum +=
      shift > 0
        ? Math.floor(total / 2 ** shift) + (Math.floor(total / 2 ** (shift - 1)) % 2)
        : total;
    chunkPrefix[chunk] = sum;
  }
  const key: [number, number] = [seed % 2 ** 32, Math.floor(seed / 2 ** 32)];
  const events: KFunctionEvent[] = [];
  for (let event = 0; event < eventCount; event++) {
    if (event >= validCount || sum === 0) {
      events.push(INACTIVE);
      continue;
    }
    const stream = new PhiloxStream(key, event, pattern, NETWORK_K_FUNCTION_STREAM);
    const chunkDraw = stream.nextBelow(sum);
    const chunk = chunkPrefix.findIndex(value => value > chunkDraw);
    const firstEdge = chunk * NETWORK_K_FUNCTION_CHUNK_EDGE_COUNT;
    const endEdge = Math.min(firstEdge + NETWORK_K_FUNCTION_CHUNK_EDGE_COUNT, edgeCount);
    const draw = stream.nextBelow(chunkTotals[chunk]);
    const fraction = (stream.nextUint32() >>> 8) / 16777216;
    const edge = firstEdge + prefix.subarray(firstEdge, endEdge).findIndex(value => value > draw);
    const weight = csr.weights[edge];
    events.push({
      first: csr.sources[edge],
      second: csr.neighbors[edge],
      firstCost: Math.fround(fraction * weight),
      secondCost: Math.fround((1 - fraction) * weight)
    });
  }
  return events;
}

/** Cumulative unordered-pair counts below each of `bandCount` thresholds (f32 thresholds, strict `<`). */
export function getPairCounts(
  csr: AccessibilityCSR,
  nodeCount: number,
  events: readonly KFunctionEvent[],
  maximumDistance: number,
  bandCount: number
): number[] {
  const thresholds = Array.from({length: bandCount}, (_, band) =>
    Math.fround(Math.fround(Math.fround(maximumDistance) * band) / (bandCount - 1))
  );
  const counts = new Array<number>(bandCount).fill(0);
  const isActive = (event: KFunctionEvent) => event.first < nodeCount && event.second < nodeCount;
  events.forEach((row, rowIndex) => {
    if (!isActive(row)) {
      return;
    }
    const costs = dijkstra(
      csr,
      nodeCount,
      [
        {node: row.first, cost: row.firstCost},
        {node: row.second, cost: row.secondCost}
      ],
      maximumDistance
    );
    for (let columnIndex = rowIndex + 1; columnIndex < events.length; columnIndex++) {
      const column = events[columnIndex];
      if (!isActive(column)) {
        continue;
      }
      let best = Infinity;
      for (const [node, cost] of [
        [column.first, column.firstCost],
        [column.second, column.secondCost]
      ]) {
        if (Number.isFinite(costs[node])) {
          best = Math.min(best, Math.fround(costs[node] + cost));
        }
      }
      if (row.first !== row.second) {
        if (row.first === column.first && row.second === column.second) {
          best = Math.min(best, Math.abs(Math.fround(row.firstCost - column.firstCost)));
        } else if (row.first === column.second && row.second === column.first) {
          best = Math.min(best, Math.abs(Math.fround(row.firstCost - column.secondCost)));
        }
      }
      thresholds.forEach((threshold, band) => {
        if (best < threshold) {
          counts[band]++;
        }
      });
    }
  });
  return counts;
}

/** `K = 2 * pairs * L / n^2`. */
export function getKValues(
  counts: readonly number[],
  validCount: number,
  networkLength: number
): number[] {
  return counts.map(count =>
    validCount >= 2 ? (2 * count * networkLength) / (validCount * validCount) : 0
  );
}
