// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';
import {fillShufflePermutation} from '../../../src/analysis-core/kernels/permutation-shuffle';
import {getPermutationSeedKey} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {getPermutedTimes, getUniquePairs, type OraclePairs} from './space-time-oracle';

/** The permutation of `[0, rows)` that `GPUKnoxTest` and the Knox family draw as number `slot`. */
export function getKnoxPermutation(rows: number, seed: number, slot: number): Uint32Array {
  const permutation = new Uint32Array(rows);
  fillShufflePermutation(
    permutation,
    rows,
    getPermutationSeedKey(seed),
    slot,
    PHILOX_PURPOSES.knoxTestShuffle.id
  );
  return permutation;
}

/** Band of `value` for strictly ascending `edges`: `(edge_k, edge_k+1]`, the first band closed at 0. */
export function getOracleBand(value: number, edges: readonly number[]): number {
  if (value < edges[0] || value > edges[edges.length - 1]) return -1;
  for (let band = 0; band < edges.length - 1; band++) {
    if (value <= edges[band + 1]) return band;
  }
  return -1;
}

/**
 * Near-repeat table in double precision by brute force over the unique unordered pairs: pair
 * counts per cell for the identity and every permutation. `distances` are the f32 values the GPU
 * is given, read as doubles.
 */
export function computeNearRepeatOracle(
  pairs: OraclePairs & {distances: ArrayLike<number>},
  times: Float32Array,
  seed: number,
  permutations: number,
  distanceEdges: readonly number[],
  timeEdges: readonly number[]
) {
  const rows = times.length;
  const unique = getUniquePairs(pairs, rows);
  const timeBands = timeEdges.length - 1;
  const cells = (distanceEdges.length - 1) * timeBands;
  const tables: number[][] = [];
  for (let slot = 0; slot <= permutations; slot++) {
    const permuted = getPermutedTimes(times, slot, seed, PHILOX_PURPOSES.knoxTestShuffle.id);
    const table = new Array<number>(cells).fill(0);
    for (let index = 0; index < unique.first.length; index++) {
      const distanceBand = getOracleBand(unique.distances[index], distanceEdges);
      const timeBand = getOracleBand(
        Math.abs(permuted[unique.first[index]] - permuted[unique.second[index]]),
        timeEdges
      );
      if (distanceBand >= 0 && timeBand >= 0) table[distanceBand * timeBands + timeBand]++;
    }
    tables.push(table);
  }
  const observed = tables[0];
  return {
    tables,
    greater: observed.map((value, cell) => tables.slice(1).filter(t => t[cell] >= value).length),
    lesser: observed.map((value, cell) => tables.slice(1).filter(t => t[cell] <= value).length),
    mean: observed.map(
      (_, cell) => tables.slice(1).reduce((sum, t) => sum + t[cell], 0) / permutations
    )
  };
}

/** Distinct, non-self neighbors of every row of a CSR, in list order. */
export function getRowNeighbors(pairs: OraclePairs, rows: number): number[][] {
  return Array.from({length: rows}, (_, row) => {
    const seen = new Set<number>();
    for (let entry = pairs.offsets[row]; entry < pairs.offsets[row + 1]; entry++) {
      const other = pairs.neighbors[entry];
      if (other !== row && other < rows) seen.add(other);
    }
    return [...seen];
  });
}

/**
 * Local Knox in double precision, written the way pointpats `_knox_local` is: for every
 * permutation and every focal event `i` build the full assignment `rids_i` (positions holding `i`
 * take the value at `i`, then position `i` keeps `i`) and count the neighbors of `i` whose assigned
 * time is within the threshold of the time of `i`. Returns per event the observed count and the
 * counts under every permutation.
 */
export function computeLocalKnoxOracle(
  pairs: OraclePairs,
  times: Float32Array,
  seed: number,
  permutations: number,
  threshold: number
) {
  const rows = times.length;
  const neighbors = getRowNeighbors(pairs, rows);
  const count = (row: number, assignment: ArrayLike<number>) =>
    neighbors[row].filter(other => Math.abs(times[row] - times[assignment[other]]) <= threshold)
      .length;
  const identity = Uint32Array.from({length: rows}, (_, row) => row);
  const observed = Array.from({length: rows}, (_, row) => count(row, identity));
  const permuted: number[][] = Array.from({length: rows}, () => []);
  for (let slot = 1; slot <= permutations; slot++) {
    const rids = getKnoxPermutation(rows, seed, slot);
    for (let row = 0; row < rows; row++) {
      const conditional = Uint32Array.from(rids);
      for (let position = 0; position < rows; position++) {
        if (rids[position] === row) conditional[position] = rids[row];
      }
      conditional[row] = row;
      permuted[row].push(count(row, conditional));
    }
  }
  return {
    observed,
    permuted,
    greater: observed.map((value, row) => permuted[row].filter(c => c >= value).length),
    lesser: observed.map((value, row) => permuted[row].filter(c => c <= value).length),
    mean: permuted.map(counts => counts.reduce((sum, c) => sum + c, 0) / permutations)
  };
}

/**
 * Spatial k-nearest-neighbor CSR in double precision: row `i` lists the `k` events nearest to `i`
 * (self excluded), nearest first. Sorting every other event by distance, with ties by row.
 */
export function getNearestNeighborCSR(positions: Float32Array, k: number) {
  const rows = positions.length / 2;
  const csr = {offsets: [0], neighbors: [] as number[], weights: [] as number[]};
  for (let row = 0; row < rows; row++) {
    const order = Array.from({length: rows}, (_, other) => other)
      .filter(other => other !== row)
      .map(other => ({
        other,
        distance: Math.hypot(
          positions[row * 2] - positions[other * 2],
          positions[row * 2 + 1] - positions[other * 2 + 1]
        )
      }))
      .sort((a, b) => a.distance - b.distance || a.other - b.other);
    for (const {other} of order.slice(0, k)) {
      csr.neighbors.push(other);
      csr.weights.push(1);
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

/**
 * Jacquez statistic in double precision, the definition of pointpats `jacquez`: build the
 * k-nearest-neighbor sets in time of the (permuted) location times by sorting every other
 * location by `|t_i - t_j|`, intersect with the spatial sets, and sum the intersection sizes over
 * the locations. The permuted time at location `r` is `times[permutation[r]]`. Times are assumed
 * free of ties at the k-th distance.
 */
export function computeJacquezOracle(
  pairs: OraclePairs,
  times: Float32Array,
  k: number,
  seed: number,
  permutations: number
): number[] {
  const rows = times.length;
  const spatial = getRowNeighbors(pairs, rows).map(list => new Set(list));
  const statistics: number[] = [];
  for (let slot = 0; slot <= permutations; slot++) {
    const permuted = getPermutedTimes(times, slot, seed, PHILOX_PURPOSES.knoxTestShuffle.id);
    let total = 0;
    for (let row = 0; row < rows; row++) {
      const nearest = Array.from({length: rows}, (_, other) => other)
        .filter(other => other !== row)
        .sort(
          (a, b) =>
            Math.abs(permuted[a] - permuted[row]) - Math.abs(permuted[b] - permuted[row]) || a - b
        )
        .slice(0, k);
      total += nearest.filter(other => spatial[row].has(other)).length;
    }
    statistics.push(total);
  }
  return statistics;
}

/**
 * Tolerance, in probability, for two independent Monte Carlo p-values of the same test:
 * `sigma * z + 1/(P + 1)` with the binomial variance at the pooled p and both permutation counts.
 */
export function getMonteCarloPValueTolerance(
  first: number,
  second: number,
  firstPermutations: number,
  secondPermutations: number,
  deviations = 4.5
): number {
  const pooled = (first + second) / 2;
  const sigma = Math.sqrt(pooled * (1 - pooled) * (1 / firstPermutations + 1 / secondPermutations));
  return deviations * sigma + 1 / (Math.min(firstPermutations, secondPermutations) + 1);
}
