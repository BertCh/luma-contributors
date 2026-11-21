// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Literal SciPy component labels from redcap-audit-b.oracle.py.
export const REDCAP_AUDIT_CASES = [
  {target: 1, tied: false, labels: [0, 0, 0, 0, 4, 4, 4, 4, 8], count: 3},
  {target: 4, tied: false, labels: [0, 0, 0, 0, 4, 4, 6, 6, 8], count: 4},
  {target: 5, tied: false, labels: [0, 0, 2, 2, 4, 4, 6, 6, 8], count: 5},
  {target: 10, tied: false, labels: [0, 1, 2, 3, 4, 5, 6, 7, 8], count: 9},
  {target: 6, tied: true, labels: [0, 0, 0, 0, 4, 5, 6, 7, 8], count: 6},
  {target: 8, tied: true, labels: [0, 0, 2, 3, 4, 5, 6, 7, 8], count: 8}
];

/** Six-row path and two islands; the path separates edge dissimilarity from region SSD. */
export function createRedcapSSDAuditInput() {
  return {
    offsets: Uint32Array.of(0, 1, 3, 5, 7, 9, 10, 10, 10),
    neighbors: Uint32Array.of(1, 0, 2, 1, 3, 2, 4, 3, 5, 4),
    weights: new Float32Array(10).fill(1),
    values: Float32Array.of(0, 1, 2.1, 3.3, 4.6, 6, 77, 88)
  };
}
