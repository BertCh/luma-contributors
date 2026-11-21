// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {flattenFeatures, type OraclePolygons} from './spatial-weights-oracle';

/** Literal pairwise-set results from audit-weights-a-contiguity.oracle.py. */
export const AUDIT_CONTIGUITY_EXPECTED = {
  queen: {
    offsets: [0, 3, 6, 9, 12, 13, 14, 14, 14],
    neighbors: [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]
  },
  rook: {
    offsets: [0, 2, 4, 6, 8, 9, 10, 10, 10],
    neighbors: [1, 2, 0, 3, 0, 3, 1, 2, 5, 4]
  }
};

export function createAuditContiguityInput(scale = 1, shift = 0) {
  const features: OraclePolygons[] = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [3, 0],
    [4, 0],
    [10, 10]
  ].map(([x, y]) => [
    [
      [
        [shift + scale * x, -shift + scale * y],
        [shift + scale * (x + 1), -shift + scale * y],
        [shift + scale * (x + 1), -shift + scale * (y + 1)],
        [shift + scale * x, -shift + scale * (y + 1)]
      ]
    ]
  ]);
  features.push([]);
  return flattenFeatures(features);
}
