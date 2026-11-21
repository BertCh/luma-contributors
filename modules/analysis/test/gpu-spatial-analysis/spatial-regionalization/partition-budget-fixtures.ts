// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export const BUDGET_ADJACENCY = [[1], [0, 2, 6, 7], [1, 3], [2, 4, 5], [3], [3], [1], [1]];
export const BUDGET_VALUES = [9, 2, 0, 2, 2, 1, 1, 9];
export const BUDGET_EXTENSIVE = [1, 1, 1, 3, 2, 3, 3, 3];

export function createPartitionInput(
  adjacency = BUDGET_ADJACENCY,
  values = BUDGET_VALUES,
  extensive = BUDGET_EXTENSIVE,
  threshold = 8
) {
  const offsets = [0];
  for (const neighbors of adjacency) offsets.push(offsets.at(-1)! + neighbors.length);
  return {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(adjacency.flat()),
    weights: new Float32Array(adjacency.flat().length).fill(1),
    values: Float32Array.from(values),
    extensiveValues: Float32Array.from(extensive),
    threshold: Float32Array.of(threshold),
    parameters: Uint32Array.of(619)
  };
}

/** Independent DFS of each induced subgraph, then direct centered SSD; no production helpers. */
export function inspectPartition(
  labels: ArrayLike<number>,
  adjacency = BUDGET_ADJACENCY,
  values = BUDGET_VALUES,
  extensive = BUDGET_EXTENSIVE
) {
  expect(labels.length).toBe(adjacency.length);
  expect(values).toHaveLength(adjacency.length);
  expect(extensive).toHaveLength(adjacency.length);
  const groups = new Map<number, number[]>();
  Array.from(labels).forEach((label, row) => {
    expect(label).not.toBe(0xffffffff);
    groups.set(label, [...(groups.get(label) ?? []), row]);
  });
  let objective = 0;
  const masses: number[] = [];
  for (const [label, rows] of groups) {
    const seen = new Set<number>();
    const pending = [rows[0]];
    while (pending.length) {
      const row = pending.pop()!;
      if (seen.has(row)) continue;
      seen.add(row);
      pending.push(
        ...adjacency[row].filter(neighbor => labels[neighbor] === label && !seen.has(neighbor))
      );
    }
    expect(
      [...seen].sort((first, second) => first - second),
      `label ${label} connectivity`
    ).toEqual(rows);
    const mean = rows.reduce((sum, row) => sum + values[row], 0) / rows.length;
    objective += rows.reduce((sum, row) => sum + (values[row] - mean) ** 2, 0);
    masses.push(rows.reduce((sum, row) => sum + extensive[row], 0));
  }
  return {objective, masses};
}

/** Max-p output contract, shared by host and device budget regressions. */
export function inspectBudgetPartition(
  labels: ArrayLike<number>,
  score: number,
  words: ArrayLike<number>,
  maximumGrowthSteps: number,
  maximumIterations: number
) {
  const actual = inspectPartition(labels);
  expect([...new Set(Array.from(labels))].sort((first, second) => first - second)).toEqual(
    Array.from({length: actual.masses.length}, (_, index) => index)
  );
  expect(words.length).toBe(6);
  expect(words[5]).toBeGreaterThanOrEqual(0);
  expect(words[5]).toBeLessThanOrEqual(maximumIterations);
  if (maximumGrowthSteps === 1 && maximumIterations === 0) {
    // Hand-checked seed-619 partition: {0}|{2}|{4}|{3,5}|{1,6}|{7}.
    // Masses 1,1,2,6,4,3; the two non-singletons each contribute (2-1)^2/2.
    expect(actual.masses).toEqual([1, 4, 1, 6, 2, 3]); // first-appearance order of labels.
    expect(actual.objective).toBe(1);
    expect(score).toBe(1);
  }
  expect(Number.isFinite(score)).toBe(true);
  expect(score).toBeCloseTo(actual.objective, 3);
  expect(words[0]).toBe(actual.masses.length);
  expect(words[2]).toBe(actual.masses.filter(mass => mass < 8).length);
  expect(words[1]).toBe(Number(actual.masses.every(mass => mass >= 8)));
  expect(words[3]).toBe(0);
  return actual;
}
