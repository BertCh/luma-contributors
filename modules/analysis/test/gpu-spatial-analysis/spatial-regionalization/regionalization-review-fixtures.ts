// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUMaxPRegionsInput} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {createAuditMaxPProps} from './regionalization-audit-max-p-fixtures';
import {createAuditAZPProps} from './regionalization-audit-azp-fixtures';

export function createReviewPath(
  values: number[],
  masses: number[],
  threshold: number,
  seed: number
): CPUMaxPRegionsInput {
  const adjacency = values.map((_, row) =>
    [row - 1, row + 1].filter(neighbor => neighbor >= 0 && neighbor < values.length)
  );
  return {
    offsets: Uint32Array.from([
      0,
      ...adjacency.map((_, row) => adjacency.slice(0, row + 1).flat().length)
    ]),
    neighbors: Uint32Array.from(adjacency.flat()),
    weights: new Float32Array(adjacency.flat().length).fill(1),
    values: Float32Array.from(values),
    extensiveValues: Float32Array.from(masses),
    threshold: Float32Array.of(threshold),
    parameters: Uint32Array.of(seed)
  };
}

export function createReviewR1() {
  return {
    input: createReviewPath([0, 1, 4], [1, 1, 2], 2, 29),
    props: {...createAuditMaxPProps(3), startCount: 1, maximumGrowthSteps: 1, maximumIterations: 5},
    labels: [0, 0, 1],
    objective: 0.5,
    status: [2, 1, 0, 0, 0, 1]
  };
}

export function createReviewR2() {
  return {
    input: {
      offsets: Uint32Array.of(0, 1, 4, 6, 7, 8),
      neighbors: Uint32Array.of(1, 0, 2, 4, 1, 3, 2, 1),
      weights: new Float32Array(8).fill(1),
      values: Float32Array.of(992.5, 1000, 1000, 1000, 999),
      initialLabels: Uint32Array.of(0, 0, 1, 1, 2),
      extensiveValues: Float32Array.of(2, 1, 1, 1, 2),
      extensiveThreshold: Float32Array.of(2),
      parameters: Uint32Array.of(0)
    },
    props: {
      ...createAuditAZPProps(),
      rowCount: 5,
      columnCount: 1,
      regionCapacity: 3,
      maximumIterations: 4
    },
    labels: [0, 2, 1, 1, 2],
    objective: 0.5,
    status: [1, 1, 0, 0]
  };
}

export function createReviewR3() {
  return {
    input: createReviewPath([1, 3, 0, 6, 0, 3], [1, 1, 1, 1, 0, 0], 2, 20),
    props: {...createAuditMaxPProps(6), startCount: 1, maximumGrowthSteps: 2},
    labels: [0, 0, 0, 0, 1, 1],
    objective: 25.5,
    status: [2, 0, 1, 0, 0, 0]
  };
}

export function createReviewR4() {
  return {
    input: createReviewPath([3, 0, 6, 0, 3, 1], [2, 1, 1, 1, 2, 2], 3, 20),
    props: {...createAuditMaxPProps(6), startCount: 2, maximumGrowthSteps: 4},
    labels: [0, 0, 1, 1, 1, 1],
    objective: 25.5,
    status: [2, 1, 0, 0, 0, 0]
  };
}
