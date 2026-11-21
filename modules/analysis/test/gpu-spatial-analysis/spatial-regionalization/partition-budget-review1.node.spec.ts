// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMaxPRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {
  createPartitionInput,
  inspectPartition,
  inspectBudgetPartition
} from './partition-budget-fixtures';

it('rejects an omitted row and non-dense Max-p labels (B4)', () => {
  expect(() =>
    inspectPartition([0, 0], [[1], [0], [3], [2]], [0, 1, 10, 11], [1, 1, 1, 1])
  ).toThrow();
  const result = computeMaxPRegionsOnCPU(createPartitionInput(), {
    standardize: false,
    maximumGrowthSteps: 8,
    maximumIterations: 0
  });
  expect(() =>
    inspectBudgetPartition(
      Array.from(result.labels, label => label + 1),
      result.objective[0],
      result.status,
      8,
      0
    )
  ).toThrow();
});

it('rejects the default-budget result masquerading as growth one (B4)', () => {
  const result = computeMaxPRegionsOnCPU(createPartitionInput(), {
    standardize: false,
    maximumGrowthSteps: 256,
    maximumIterations: 0
  });
  expect(() =>
    inspectBudgetPartition(result.labels, result.objective[0], result.status, 1, 0)
  ).toThrow();
});

it('rejects an improvement-count word beyond the requested budget (B4)', () => {
  const result = computeMaxPRegionsOnCPU(createPartitionInput(), {
    standardize: false,
    maximumGrowthSteps: 8,
    maximumIterations: 0
  });
  const words = result.status.slice();
  words[5] = 1;
  expect(() => inspectBudgetPartition(result.labels, result.objective[0], words, 8, 0)).toThrow();
});
