// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Binary64 CPU reference for GPUExpectationBasedSpaceTimeScan: the exact best subset of every
// center's nearest-zone neighborhood by exhaustive enumeration (up to `2^k` subsets, independent of
// the kernel's sort-and-prefix-scan), and the Poisson replicates reproduced from the same Philox
// streams. The `scanstatistics` R package (GPL, R-only) was not run; the external fixture is pending.

import {PhiloxStream} from '../../../src/analysis-core/kernels/permutation-random';
import {getPhiloxOpenUnitFloat} from '../../../src/analysis-core/kernels/philox-random';

/** Philox purpose tag of the scan's replicate streams, equal to the contributor's (`'EBCS'`). */
const SCAN_STREAM = 0x45424353;

export type ExpectationBasedScanScene = {
  positions: Float32Array;
  cases: Uint32Array;
  baseline: Float32Array;
  timeBuckets: number;
  maximumNeighbors: number;
  maximumWindowLength: number;
};

/** The expectation-based Poisson score `C ln(C/B) + B - C` for `C > B`, else 0. */
export function getExpectationBasedScore(cases: number, baseline: number): number {
  if (!(cases > baseline) || !(baseline > 0)) return 0;
  const delta = (cases - baseline) / baseline;
  // (1+d) log(1+d)-d = d²/2-d³/6+d⁴/12-d⁵/20+..., without cancellation near equality.
  if (delta < 1e-3) {
    return (
      baseline *
      delta *
      delta *
      (0.5 - delta / 6 + (delta * delta) / 12 - delta ** 3 / 20 + delta ** 4 / 30)
    );
  }
  return cases * Math.log1p(delta) + baseline - cases;
}

function getNearestZones(scene: ExpectationBasedScanScene, center: number): number[] {
  const zones = scene.positions.length / 2;
  const centerX = scene.positions[center * 2];
  const centerY = scene.positions[center * 2 + 1];
  return Array.from({length: zones}, (_, zone) => zone)
    .map(zone => ({
      zone,
      distance:
        (scene.positions[zone * 2] - centerX) ** 2 + (scene.positions[zone * 2 + 1] - centerY) ** 2
    }))
    .sort((left, right) => left.distance - right.distance || left.zone - right.zone)
    .slice(0, scene.maximumNeighbors)
    .map(entry => entry.zone);
}

/** Best subset of each center's neighborhood over windows ending at the last bucket, by enumeration. */
export function computeExpectationBasedScanReference(
  scene: ExpectationBasedScanScene,
  windowLimit: number
) {
  const zones = scene.positions.length / 2;
  if (Math.min(zones, scene.maximumNeighbors) > 16) {
    throw new Error(
      'expectation scan exhaustive reference supports at most 16 neighbors; use an independent closed-form fixture for larger neighborhoods'
    );
  }
  const {timeBuckets, cases, baseline} = scene;
  const centerScores = new Float64Array(zones);
  let best = {
    score: 0,
    center: -1,
    windowLength: 0,
    members: [] as number[],
    observed: 0,
    expected: 0
  };
  for (let center = 0; center < zones; center++) {
    const neighborhood = getNearestZones(scene, center);
    for (let windowLength = 1; windowLength <= windowLimit; windowLength++) {
      const caseSums = neighborhood.map(zone => {
        let sum = 0;
        for (let bucket = timeBuckets - windowLength; bucket < timeBuckets; bucket++) {
          const expected = baseline[zone * timeBuckets + bucket];
          if (Number.isFinite(expected) && expected > 0) sum += cases[zone * timeBuckets + bucket];
        }
        return sum;
      });
      const baselineSums = neighborhood.map(zone => {
        let sum = 0;
        for (let bucket = timeBuckets - windowLength; bucket < timeBuckets; bucket++) {
          const expected = baseline[zone * timeBuckets + bucket];
          if (Number.isFinite(expected) && expected > 0) sum += expected;
        }
        return sum;
      });
      for (let mask = 1; mask < 1 << neighborhood.length; mask++) {
        let observed = 0;
        let expected = 0;
        let valid = true;
        for (let slot = 0; slot < neighborhood.length; slot++) {
          if (mask & (1 << slot)) {
            if (!(baselineSums[slot] > 0)) {
              valid = false;
              break;
            }
            observed += caseSums[slot];
            expected += baselineSums[slot];
          }
        }
        if (!valid) continue;
        const score = getExpectationBasedScore(observed, expected);
        centerScores[center] = Math.max(centerScores[center], score);
        if (score > best.score) {
          best = {
            score,
            center,
            windowLength,
            members: neighborhood.filter((_, slot) => mask & (1 << slot)).sort((a, b) => a - b),
            observed,
            expected
          };
        }
      }
    }
  }
  return {best, centerScores};
}

/**
 * Knuth's Poisson draw with f32 rounding, for means below 30. The uniforms are the library's
 * open-interval Philox uniform `(0, 1)` (the CPU mirror of `philoxOpenUnitFloat`).
 */
function drawPoissonReference(stream: PhiloxStream, mean: number): number {
  if (!(mean > 0) || !Number.isFinite(mean)) return 0;
  const limit = Math.fround(Math.exp(-mean));
  let product = 1;
  let count = 0;
  for (;;) {
    product = Math.fround(product * getPhiloxOpenUnitFloat(stream.nextUint32()));
    if (product <= limit) break;
    count++;
    if (count >= 1024) break;
  }
  return count;
}

/** Replicate maxima for `replicateCount` draws of the kernel's Philox streams (means below 30 only). */
export function computeReplicateMaximaReference(
  scene: ExpectationBasedScanScene,
  seed: number,
  replicateCount: number,
  windowLimit: number
): Float64Array {
  const zones = scene.positions.length / 2;
  const {timeBuckets, baseline, maximumWindowLength, maximumNeighbors} = scene;
  const windowCells = zones * maximumWindowLength;
  const maxima = new Float64Array(replicateCount);
  const neighborhoods = Array.from({length: zones}, (_, center) => getNearestZones(scene, center));
  const key: [number, number] = [seed % 2 ** 32, Math.floor(seed / 2 ** 32)];
  for (let replicate = 0; replicate < replicateCount; replicate++) {
    const counts = new Uint32Array(windowCells);
    for (let windowCell = 0; windowCell < windowCells; windowCell++) {
      const zone = Math.floor(windowCell / maximumWindowLength);
      const windowBucket = windowCell % maximumWindowLength;
      const mean = baseline[zone * timeBuckets + timeBuckets - maximumWindowLength + windowBucket];
      counts[windowCell] = drawPoissonReference(
        new PhiloxStream(key, replicate, windowCell, SCAN_STREAM),
        mean
      );
    }
    let maximum = 0;
    for (let center = 0; center < zones; center++) {
      const neighborhood = neighborhoods[center];
      for (let windowLength = 1; windowLength <= windowLimit; windowLength++) {
        const rows = neighborhood
          .map(zone => {
            let observed = 0;
            let expected = 0;
            for (let back = 0; back < windowLength; back++) {
              const cellBaseline = baseline[zone * timeBuckets + timeBuckets - 1 - back];
              if (Number.isFinite(cellBaseline) && cellBaseline > 0) {
                observed += counts[zone * maximumWindowLength + maximumWindowLength - 1 - back];
                expected += cellBaseline;
              }
            }
            return {observed, expected};
          })
          .filter(row => row.expected > 0)
          .sort((left, right) => right.observed / right.expected - left.observed / left.expected);
        let observed = 0;
        let expected = 0;
        for (const row of rows) {
          observed += row.observed;
          expected += row.expected;
          maximum = Math.max(maximum, getExpectationBasedScore(observed, expected));
        }
      }
    }
    maxima[replicate] = maximum;
  }
  void maximumNeighbors;
  return maxima;
}
