// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Two paths of four rows and an island. Only canonical (lower-row) slots mark tree edges.
export function createAuditForest() {
  return {
    offsets: Uint32Array.of(0, 1, 3, 5, 6, 7, 9, 11, 12, 12),
    neighbors: Uint32Array.of(1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6),
    weights: new Float32Array(12).fill(1),
    treeEdgeFlags: Uint32Array.of(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
    componentLabels: Uint32Array.of(0, 0, 0, 0, 4, 4, 4, 4, 8),
    values: Float32Array.of(0, 0, 0, 0, 10, 2, 10, 2, 1, -1, 1, -1, 5, 3, 5, 3, 99, 7),
    parameters: Uint32Array.of(6, 2)
  };
}

export function createAuditWardInput() {
  return {
    ...createAuditForest(),
    values: Float32Array.of(0, 2, 10, 14, 1, 3, 20, 26, 99),
    parameters: Uint32Array.of(0)
  };
}
