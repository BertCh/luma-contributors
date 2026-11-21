// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const BOUNDED_ALLOCATION_SCENES = [
  {
    operation: 'p-center' as const,
    costs: [1, 5, 9, 9, 2, 6, 8, 7, 1],
    demand: [1, 1, 1],
    capacities: [3, 3, 3],
    // All subsets: 01 -> costs [1,2,7], max 7; 02 -> [1,6,1], max 6;
    // 12 -> [5,2,1], max 5. Every subset is feasible; only 12 is optimal.
    selected: [0, 1, 1],
    assignments: [1, 1, 2],
    objective: 5,
    status: [0, 1, 1],
    feasibleWorseObjective: 6
  },
  {
    operation: 'capacitated-p-median' as const,
    costs: [7, 3, 6, 5, 7, 2, 4, 2, 7],
    demand: [2, 1, 2],
    capacities: [3, 3, 3],
    // Demand-order nearest-feasible assignment, all three subsets hand-enumerated:
    // 01 -> assignments [1,0,0], weighted cost 2*3+5+2*4=19;
    // 02 -> [2,2,0], cost 22; 12 -> [1,2,2], cost 22.
    // Ignoring capacity chooses 12 at cost 12; globally optimizing assignments
    // also chooses 12, at cost 18. Both violate this documented assignment rule.
    // All are feasible. This is an optimum under the documented greedy assignment rule;
    // the status must still decline the global capacitated-assignment certificate.
    selected: [1, 1, 0],
    assignments: [1, 0, 0],
    objective: 19,
    status: [0, 1, 0],
    feasibleWorseObjective: 22
  }
];
