// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computePycnophylacticDetailedOracle} from './areal-interpolation-oracle';

// This is the existing test-side host reference, not a production CPU twin.
// Independent hand calculation: starting [1,1,4], rook averaging plus zone-total
// restoration gives [1/4,7/4,4], then [13/16,19/16,4]. Largest relative changes
// are 3/4 and 9/16. There is no clipping; the multiplicative correction is 1.
// Doubled represented totals yield [13/8,19/8,8] at step 2: absolute change
// 9/8 divided by mean 2 equals 9/16. Equality must converge (inclusive relative).
for (const testCase of [
  {budget: 0, tolerance: 0.6, status: [0, 0, 1], mass: [1, 1, 4], changes: []},
  {budget: 1, tolerance: 0.6, status: [0, 1, 1], mass: [0.25, 1.75, 4], changes: [0.75]},
  {
    budget: 2,
    tolerance: 0.5,
    status: [0, 2, 1],
    mass: [0.8125, 1.1875, 4],
    changes: [0.75, 0.5625]
  },
  {
    budget: 2,
    tolerance: 9 / 16,
    totals: [4, 8, 3],
    status: [1, 2, 1],
    mass: [1.625, 2.375, 8],
    changes: [0.75, 0.5625]
  },
  {
    budget: 4,
    tolerance: 0.6,
    status: [1, 2, 1],
    mass: [0.8125, 1.1875, 4],
    changes: [0.75, 0.5625]
  }
]) {
  it(`host reference status follows the hand-derived stopping decision (budget ${testCase.budget}, tolerance ${testCase.tolerance})`, () => {
    for (const zones of [
      [0, 0, 1],
      [0, 2, 1],
      [0, 0, 1]
    ]) {
      const result = computePycnophylacticDetailedOracle({
        width: 3,
        height: 1,
        zones,
        zoneCount: 3,
        totals: testCase.totals ?? [2, 4, 3],
        iterations: testCase.budget,
        kernel: 'rook',
        tolerance: testCase.tolerance
      });
      const singletonZones = zones[1] === 2;
      expect([Number(result.converged), result.changes.length, result.invalidCount]).toEqual(
        singletonZones ? (testCase.budget === 0 ? [0, 0, 0] : [1, 1, 0]) : testCase.status
      );
      expect(result.mass).toEqual(
        singletonZones ? (testCase.totals ? [4, 3, 8] : [2, 3, 4]) : testCase.mass
      );
      expect(result.changes).toEqual(
        singletonZones ? (testCase.budget === 0 ? [] : [0]) : testCase.changes
      );
      expect(result.convergedAfter).toBe(result.converged ? (singletonZones ? 1 : 2) : undefined);
    }
  });
}
