// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMaxPRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {computeAZPRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {
  createPartitionInput,
  inspectPartition,
  inspectBudgetPartition
} from './partition-budget-fixtures';

for (let maximumGrowthSteps = 1; maximumGrowthSteps <= 8; maximumGrowthSteps++) {
  for (const maximumIterations of [0, 4]) {
    it(`keeps seed 619 regions contiguous at growth ${maximumGrowthSteps}, improvement ${maximumIterations}`, () => {
      const result = computeMaxPRegionsOnCPU(createPartitionInput(), {
        standardize: false,
        maximumGrowthSteps,
        maximumIterations
      });
      if (maximumGrowthSteps === 2) expect(Array.from(result.evidence.slice(0, 2))).toEqual([1, 0]);
      expect(result.evidence[2]).toBe(Number(maximumIterations > 0));
      inspectBudgetPartition(
        result.labels,
        result.objective[0],
        result.status,
        maximumGrowthSteps,
        maximumIterations
      );
    });
  }
}

it('attains the enumerable two-region optimum on a four-row path in Max-p and AZP', () => {
  const adjacency = [[1], [0, 2], [1, 3], [2]];
  const values = [0, 1, 10, 11];
  const input = createPartitionInput(adjacency, values, [1, 1, 1, 1], 2);
  // Threshold two permits at most two regions. The only contiguous two-region partition
  // with both masses >=2 is {0,1}|{2,3}, SSD=(1/2)+(1/2)=1. One region has SSD 101.
  const maximum = computeMaxPRegionsOnCPU(input, {standardize: false, maximumIterations: 4});
  const azp = computeAZPRegionsOnCPU(
    {...input, initialLabels: Uint32Array.of(0, 1, 1, 1), extensiveThreshold: input.threshold},
    {standardize: false, regionCapacity: 2, maximumIterations: 4}
  );
  for (const result of [maximum, azp]) {
    const actual = inspectPartition(result.labels, adjacency, values, [1, 1, 1, 1]);
    expect(actual.masses).toEqual([2, 2]);
    expect(actual.objective).toBe(1);
    expect(Number.isFinite(result.objective[0])).toBe(true);
    expect(result.objective[0]).toBe(1);
  }
});
