// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demand-major 3 x 4 matrix. All six two-facility objective values are
// {0,1}:7, {0,2}:4, {0,3}:2, {1,2}:4, {1,3}:1, {2,3}:4.
// Single-facility objectives are 12,13,7,8, so greedy picks 2 then 0 (tie with 1/3).
// First improving best swap gives {0,3}, cost 2. The named unresolved next swap
// replaces 0 by 1, giving cost 1. Every neighbor of {1,3} is worse, certifying stop.
export const ALLOCATION_STATUS_COSTS = [1, 7, 2, 0, 2, 1, 4, 8, 9, 5, 1, 0];
export const ALLOCATION_STATUS_CASES = [
  {
    maximumSwapIterations: 0,
    costs: ALLOCATION_STATUS_COSTS,
    status: [2, 0, 1, 0, 0, 0, 0],
    selected: [1, 0, 1, 0],
    objective: 4
  },
  {
    maximumSwapIterations: 1,
    costs: ALLOCATION_STATUS_COSTS,
    status: [2, 0, 1, 0, 1, 0, 1],
    selected: [1, 0, 0, 1],
    objective: 2
  },
  {
    maximumSwapIterations: 3,
    costs: ALLOCATION_STATUS_COSTS,
    status: [2, 0, 1, 0, 2, 1, 0],
    selected: [0, 1, 0, 1],
    objective: 1
  },
  // Two invalid entries in the same demand row: the diagnostic counts entries,
  // not rows and not merely presence. Validation aborts before any selection.
  {
    maximumSwapIterations: 2,
    costs: [-1, NaN, ...ALLOCATION_STATUS_COSTS.slice(2)],
    status: [0, 3, 0, 2, 0, 0, 0],
    selected: [0, 0, 0, 0],
    objective: 0
  }
];
