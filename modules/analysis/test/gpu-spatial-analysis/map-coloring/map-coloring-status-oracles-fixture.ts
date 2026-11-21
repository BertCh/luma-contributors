// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Degree, not a copied priority hash, fixes the order. In the symmetric two-row
// graph row 1 has a self slot, hence degree 2 > row 0's degree 1. Separate GPU
// chunks execute row 0 before row 1: the first round must leave row 0 blocked.
// With a second round row 0 takes color 1. Self slots do not conflict.
// The asymmetric graph has row 0 -> row 1 but no reverse edge. Row 0 (degree 2)
// colors first; row 1 has no neighbors and also takes 0, forcing one conflict.
export const MAP_COLORING_STATUS_CASES = [
  {
    name: 'isolate',
    offsets: [0, 0],
    neighbors: [] as number[],
    maximumIterations: 1,
    colors: [0],
    colorCount: 1,
    conflictCount: 0,
    converged: 1,
    iterationCount: 1
  },
  {
    name: 'blocked first chunk',
    offsets: [0, 1, 3],
    neighbors: [1, 0, 1],
    maximumIterations: 1,
    colors: [4294967295, 0],
    colorCount: 1,
    conflictCount: 0,
    converged: 0,
    iterationCount: 1
  },
  {
    name: 'second round completes',
    offsets: [0, 1, 3],
    neighbors: [1, 0, 1],
    maximumIterations: 2,
    colors: [1, 0],
    colorCount: 2,
    conflictCount: 0,
    converged: 1,
    iterationCount: 2
  },
  {
    name: 'asymmetric conflict diagnostic',
    offsets: [0, 2, 2],
    neighbors: [0, 1],
    maximumIterations: 1,
    colors: [0, 0],
    colorCount: 1,
    conflictCount: 1,
    converged: 1,
    iterationCount: 1
  },
  {
    // Row 0 waits for degree-2 row 2. Row 1 takes 0, row 2 takes 1,
    // then row 0 takes 0 on round 2. Equal-color entry 1->0 is descending
    // and excluded by the diagnostic's neighbor > row counting convention.
    name: 'descending asymmetric edge is not counted',
    offsets: [0, 1, 3, 5],
    neighbors: [2, 0, 1, 1, 2],
    maximumIterations: 2,
    colors: [0, 0, 1],
    colorCount: 2,
    conflictCount: 0,
    converged: 1,
    iterationCount: 2
  }
];
