// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
export const costTableOracleInput = {
  topology: {
    offsets: Uint32Array.of(0, 1, 2, 2),
    neighbors: Uint32Array.of(1, 2),
    weights: Float32Array.of(2, 3)
  },
  costLimit: Float32Array.of(5)
};
// Enumerating all downstream vertices, including the diagonal, yields rows
// [(0,0),(1,2),(2,5)], [(1,0),(2,3)], [(2,0)]. A single hop omits only (0,2).
export function getCostTableExpected(stopped: boolean) {
  return stopped
    ? {offsets: [0, 2, 4, 5], targets: [0, 1, 1, 2, 2], costs: [0, 2, 0, 3, 0], converged: [0]}
    : {
        offsets: [0, 3, 5, 6],
        targets: [0, 1, 2, 1, 2, 2],
        costs: [0, 2, 5, 0, 3, 0],
        converged: [1]
      };
}
