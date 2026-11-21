// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUSpatialPredicateJoinExecutionPlan} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-route-plan';

it('forward N1 reports the auto crossover independently of route feasibility', () => {
  const shape = {
    predicate: 'intersects' as const,
    leftKind: 'lines' as const,
    rightKind: 'lines' as const,
    leftCount: 3,
    rightCount: 2,
    leftVertexCount: 9,
    rightVertexCount: 12
  };
  // Hand calculation: (9 / 3) * (12 / 2) = 18 < 1024.
  const small = getGPUSpatialPredicateJoinExecutionPlan(shape);
  expect(small.route).toBe('fast');
  expect(small.feasible).toBe(true);
  expect(small.constraints[0]).toMatchObject({minimum: 1024, received: 18, satisfied: false});
  // 32 vertices per feature on each side gives the inclusive crossover 32 * 32 = 1024.
  const threshold = getGPUSpatialPredicateJoinExecutionPlan({
    ...shape,
    leftVertexCount: 96,
    rightVertexCount: 64
  });
  expect(threshold.route).toBe('relate');
  expect(threshold.feasible).toBe(true);
  expect(threshold.constraints[0]).toMatchObject({minimum: 1024, received: 1024, satisfied: true});
});
