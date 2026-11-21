// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUMaxPRegionsInput} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import type {MaxPRegionsOperationProps} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation';

export function createAuditMaxPInput(island = false): CPUMaxPRegionsInput {
  return {
    offsets: Uint32Array.from(
      island ? [0, 1, 1, 2, 2, 3, 3, 4, 4, 4] : [0, 1, 1, 2, 2, 3, 3, 4, 4]
    ),
    neighbors: Uint32Array.of(1, 3, 5, 7),
    weights: Float32Array.of(0, -3, 4, 7),
    values: Float32Array.from([0, 2, 10, 14, -4, -4, 20, 26, ...(island ? [99] : [])]),
    extensiveValues: Float32Array.from([1, 2, 1, 2, 1, 2, 1, 2, ...(island ? [1] : [])]),
    threshold: Float32Array.of(3),
    parameters: Uint32Array.of(0)
  };
}

export function createAuditMaxPProps(rowCount = 8): MaxPRegionsOperationProps {
  return {
    rowCount,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    values: 'values',
    standardize: false,
    extensiveValues: 'extensiveValues',
    threshold: 'threshold',
    startCount: 4,
    maximumGrowthSteps: 64,
    parameters: 'parameters',
    output: {labels: 'labels', objective: 'objective', status: 'status'}
  };
}

export function createAuditMaxPSingletons(): CPUMaxPRegionsInput {
  return {
    offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8),
    neighbors: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7),
    weights: new Float32Array(8).fill(1),
    values: Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7),
    extensiveValues: new Float32Array(8).fill(1),
    threshold: Float32Array.of(1),
    parameters: Uint32Array.of(0)
  };
}

/** Fixed six-by-six rook lattice: a four-round enclave budget can remove an articulation. */
export function createAuditMaxPRemainderInput(): CPUMaxPRegionsInput {
  const adjacency = Array.from({length: 36}, (_, row) =>
    [row - 6, row + 6, ...(row % 6 > 0 ? [row - 1] : []), ...(row % 6 < 5 ? [row + 1] : [])].filter(
      neighbor => neighbor >= 0 && neighbor < 36
    )
  );
  return {
    offsets: Uint32Array.from([
      0,
      ...adjacency.map((_, row) => adjacency.slice(0, row + 1).flat().length)
    ]),
    neighbors: Uint32Array.from(adjacency.flat()),
    weights: new Float32Array(adjacency.flat().length).fill(1),
    values: Float32Array.of(
      0,
      12,
      14,
      6,
      7,
      17,
      17,
      7,
      6,
      14,
      12,
      0,
      16,
      3,
      18,
      4,
      18,
      3,
      16,
      0,
      12,
      14,
      6,
      7,
      17,
      17,
      7,
      6,
      14,
      12,
      0,
      16,
      3,
      18,
      4,
      18
    ),
    extensiveValues: Float32Array.of(
      2,
      1,
      0,
      3,
      2,
      1,
      0,
      3,
      2,
      1,
      0,
      3,
      2,
      1,
      0,
      3,
      2,
      1,
      0,
      3,
      2,
      1,
      0,
      3,
      2,
      1,
      0,
      3,
      2,
      1,
      0,
      3,
      2,
      1,
      0,
      3
    ),
    threshold: Float32Array.of(7),
    parameters: Uint32Array.of(14)
  };
}

/** Independent definitions: BFS connectivity, mass floors and pairwise SSD / region size.
 * The heuristic is free to choose its partition; these definitions apply to any published one.
 */
export function evaluateAuditMaxPRemainder(labels: ArrayLike<number>) {
  const input = createAuditMaxPRemainderInput();
  const groups = new Map<number, number[]>();
  for (let row = 0; row < 36; row++)
    groups.set(labels[row], [...(groups.get(labels[row]) ?? []), row]);
  let connected = true;
  let objective = 0;
  let infeasibleRegionCount = 0;
  for (const [label, members] of groups) {
    const reached = new Set([members[0]]);
    for (const row of reached)
      for (let slot = input.offsets[row]; slot < input.offsets[row + 1]; slot++) {
        const neighbor = input.neighbors[slot];
        if (labels[neighbor] === label) reached.add(neighbor);
      }
    connected &&= reached.size === members.length;
    infeasibleRegionCount += Number(
      members.reduce((sum, row) => sum + input.extensiveValues[row], 0) < 7
    );
    // Huygens' identity: SSD = sum of squared distances over unordered pairs / group size.
    for (let first = 0; first < members.length; first++)
      for (let second = first + 1; second < members.length; second++)
        objective +=
          (input.values[members[first]] - input.values[members[second]]) ** 2 / members.length;
  }
  return {connected, objective, infeasibleRegionCount, regionCount: groups.size};
}
