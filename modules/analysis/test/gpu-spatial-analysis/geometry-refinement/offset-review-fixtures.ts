// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const OFFSET_REVIEW_CASES = [
  ...[0, 10000000].map(origin => ({
    name: `covered horizontal legs translated by ${origin}`,
    original: [origin, 0, origin + 60, 0, origin + 60, 3, origin, 3],
    raw: [origin, 2, origin + 58, 2, origin + 58, 1, origin, 1],
    distance: 2,
    expected: [] as number[]
  })),
  {
    name: 'integer endpoint tangencies at distance 25',
    original: [24, -7, 31, 17],
    raw: [0, 0, 7, 24],
    distance: 25,
    // Direction (7,24) has length 25; (-24,7) is an exactly perpendicular offset.
    expected: [0, 0, 7, 24]
  },
  {
    name: 'integer endpoint tangencies at distance 5',
    original: [0, 0, 15, 20],
    raw: [-4, 3, 11, 23],
    distance: 5,
    // Direction (15,20) has length 25; (-4,3) is an exactly perpendicular offset.
    expected: [-4, 3, 11, 23]
  }
];
