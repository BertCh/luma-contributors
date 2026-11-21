// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSkaterRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {computeWardRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {createAuditForest, createAuditWardInput} from './regionalization-audit-twins-fixtures';

it('audits every SKATER twin output with NumPy SSD differences, slot ties and an island', () => {
  for (const [scale, shift] of [
    [1, 0],
    [3, 64],
    [0, 0]
  ]) {
    const input = createAuditForest();
    input.values = input.values.map(value => scale * value + shift);
    const result = computeSkaterRegionsOnCPU(input, {columnCount: 2, maximumRegionCount: 6});
    expect(Array.from(result.labels)).toEqual([0, 0, 2, 2, 4, 4, 6, 6, 8]);
    expect(Array.from(result.regionCount)).toEqual([5]);
    expect(Array.from(result.cutEdges)).toEqual([2, 8, 4294967295, 4294967295, 4294967295]);
    expect(Array.from(result.cutGains)).toEqual([104 * scale ** 2, 32 * scale ** 2, 0, 0, 0]);
  }
});

it('audits both Ward twin schedules and round budgets using direct NumPy SSD differences', () => {
  for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
    for (const [scale, shift, standardize, expected] of [
      [1, 0, false, 30],
      [3, 64, false, 270],
      [1, 0, true, 0.0346855462616689]
    ] as const) {
      const input = createAuditWardInput();
      input.values = input.values.map(value => scale * value + shift);
      const result = computeWardRegionsOnCPU(input, {
        targetRegionCount: 5,
        maximumIterations: 4,
        mergeSchedule,
        standardize,
        statusLength: 5
      });
      expect(Array.from(result.labels)).toEqual([0, 0, 2, 2, 4, 4, 6, 6, 8]);
      expect(Array.from(result.status)).toEqual([
        4,
        1,
        0,
        0,
        mergeSchedule === 'sequential' ? 0 : 1
      ]);
      expect(result.objective[0]).toBe(Math.fround(expected));
    }
  }
  const limited = computeWardRegionsOnCPU(createAuditWardInput(), {
    targetRegionCount: 5,
    maximumIterations: 3,
    standardize: false
  });
  expect(Array.from(limited.labels)).toEqual([0, 0, 2, 2, 4, 4, 6, 7, 8]);
  expect(Array.from(limited.status)).toEqual([3, 0, 1, 0]);
  expect(Array.from(limited.objective)).toEqual([12]);
  // Two four-row components and an island cannot reach two regions. Their direct component
  // SSDs are 131 and 461; an exhausted graph is distinct from an iteration limit.
  const disconnected = computeWardRegionsOnCPU(createAuditWardInput(), {
    targetRegionCount: 2,
    maximumIterations: 8,
    standardize: false
  });
  expect(Array.from(disconnected.labels)).toEqual([0, 0, 0, 0, 4, 4, 4, 4, 8]);
  expect(Array.from(disconnected.status)).toEqual([6, 0, 0, 0]);
  expect(Array.from(disconnected.objective)).toEqual([592]);
});
