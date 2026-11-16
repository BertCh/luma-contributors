// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OffsetCurveReferenceCase} from './offset-curve-reference';

/** Ordered GEOS sequence with explicitly pinned GPU-only first outside corners. */
export function getCompactOffsetExpected(reference: OffsetCurveReferenceCase): number[][] {
  let expected = reference.expected.map(point => [...point]);
  if (!reference.ring) return expected;
  if (reference.name === 'ring-mitre--0.5') {
    expected = [[-0.5, -0.5], ...expected];
  } else if (reference.name === 'ring-round--0.5') {
    // A quarter-circle centered at (0, 0), radius 0.5, in input-corner order.
    // GEOS omits this first outside arc; these values pin the entire additional corner.
    expected = [
      [-0.5, 0],
      [-0.49039264, -0.097545161],
      [-0.461939766, -0.191341716],
      [-0.415734806, -0.277785117],
      [-0.353553391, -0.353553391],
      [-0.277785117, -0.415734806],
      [-0.191341716, -0.461939766],
      [-0.097545161, -0.49039264],
      [0, -0.5],
      ...expected.slice(1)
    ];
  }
  const first = expected[0];
  const last = expected[expected.length - 1];
  if (first[0] === last[0] && first[1] === last[1]) expected.pop();
  return expected;
}
