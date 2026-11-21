// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const FACILITY_SWAP_TIE_SCENES = [
  {
    name: 'greedy tie and zero-gain rejection',
    costs: [1, 1, 1, 1, 1, 1],
    demandCount: 2,
    facilityCount: 3,
    maximumFacilityCount: 1,
    maximumSwapIterations: 2,
    greedySelected: [0],
    selected: [1, 0, 0],
    assignments: [0, 0],
    objective: 2,
    status: [1, 0, 1, 0, 0, 1, 0]
  },
  {
    name: 'strictly improving swap tie',
    costs: [9, 8, 7, 4, 4, 8, 9, 8, 5, 1, 2, 2, 7, 5, 10, 10, 11, 10, 6, 5],
    demandCount: 4,
    facilityCount: 5,
    maximumFacilityCount: 2,
    maximumSwapIterations: 1,
    // Greedy {3,4}: 4+1+5+5=15. Replacing 3 by 0 or 1 gives 4+1+2+5=12.
    // Lowest removed facility then lowest added candidate selects {0,4}.
    greedySelected: [3, 4],
    selected: [1, 0, 0, 0, 1],
    assignments: [4, 4, 0, 4],
    objective: 12,
    status: [2, 0, 1, 0, 1, 0, 1]
  }
];
