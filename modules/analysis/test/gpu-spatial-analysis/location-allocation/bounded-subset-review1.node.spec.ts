// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {BOUNDED_ALLOCATION_SCENES} from './bounded-subset-fixtures';

/** Exhaust all three subsets and all 2^3 assignments; independent of allocation kernels. */
it('uses a capacitated fixture that distinguishes capacity and demand-order assignment (B3)', () => {
  const scene = BOUNDED_ALLOCATION_SCENES[1];
  const subsets = [
    [0, 1],
    [0, 2],
    [1, 2]
  ];
  const capacityScores: number[] = [],
    unlimitedScores: number[] = [],
    optimalScores: number[] = [];
  const greedyAssignments: number[][] = [];
  for (const subset of subsets) {
    const remaining = [...scene.capacities];
    const assignments = scene.demand.map((amount, row) => {
      const options = subset
        .filter(facility => remaining[facility] >= amount)
        .sort(
          (first, second) =>
            scene.costs[row * 3 + first] - scene.costs[row * 3 + second] || first - second
        );
      const chosen = options[0];
      if (chosen === undefined) return 0xffffffff;
      remaining[chosen] -= amount;
      return chosen;
    });
    const cost = (labels: number[]) =>
      labels.reduce(
        (sum, facility, row) => sum + scene.demand[row] * scene.costs[row * 3 + facility],
        0
      );
    greedyAssignments.push(assignments);
    capacityScores.push(cost(assignments));
    unlimitedScores.push(
      scene.demand.reduce(
        (sum, amount, row) =>
          sum + amount * Math.min(...subset.map(facility => scene.costs[row * 3 + facility])),
        0
      )
    );
    let best = Infinity;
    for (let bits = 0; bits < 8; bits++) {
      const labels = [0, 1, 2].map(row => subset[(bits >> row) & 1]);
      if (
        subset.every(
          facility =>
            scene.demand.reduce(
              (sum, amount, row) => sum + (labels[row] === facility ? amount : 0),
              0
            ) <= scene.capacities[facility]
        )
      )
        best = Math.min(best, cost(labels));
    }
    optimalScores.push(best);
  }
  expect(capacityScores).toEqual([19, 22, 22]);
  expect(greedyAssignments[0]).toEqual([1, 0, 0]);
  expect(Math.min(...unlimitedScores)).toBe(12);
  expect(unlimitedScores.indexOf(12)).toBe(2);
  expect(Math.min(...optimalScores)).toBe(18);
  expect(optimalScores.indexOf(18)).toBe(2);
});
