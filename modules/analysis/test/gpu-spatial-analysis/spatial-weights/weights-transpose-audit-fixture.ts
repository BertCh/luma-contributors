// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent SciPy sparse transpose: weights-transpose-audit.oracle.py.
export const TRANSPOSE_AUDIT_INPUT = {
  offsets: [0, 3, 5, 8, 9, 10, 11, 12, 13],
  neighbors: [0, 1, 2, 0, 2, 0, 1, 3, 2, 5, 4, 7, 9],
  weights: [0.5, 2, 1, 1, 3, 1, 3, 4, 0, 2, 2, 0, 99],
  distances: [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5]
};

export const TRANSPOSE_AUDIT_EXPECTED = {
  offsets: [0, 3, 5, 8, 9, 10, 11, 11, 12],
  neighbors: [0, 1, 2, 0, 2, 0, 1, 3, 2, 5, 4, 6, 0, 0, 0],
  weights: [0.5, 1, 1, 2, 3, 1, 3, 0, 4, 2, 2, 0, 0, 0, 0],
  distances: [10.5, 13.5, 15.5, 11.5, 16.5, 12.5, 14.5, 18.5, 17.5, 20.5, 19.5, 21.5, 0, 0, 0]
};
