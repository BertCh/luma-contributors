// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Two edge-adjacent squares: four queen candidate pairs deduplicate to two directed links. */
export function createPairStarvedContiguityFixture() {
  return {
    input: {
      positions: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1]),
      ringOffsets: new Uint32Array([0, 4, 8]),
      polygonOffsets: new Uint32Array([0, 1, 2]),
      featureOffsets: new Uint32Array([0, 1, 2])
    },
    options: {criterion: 'queen' as const, capacity: 8, pairCapacity: 1}
  };
}

/** Three rectangles: 160 shared subdivisions exhaust the explorer's 288-pair budget. */
export function createSubdividedContiguityFixture() {
  const subdivisions = 160;
  const first = [
    [0, 0],
    [1, 0],
    ...Array.from({length: subdivisions}, (_, index) => [1, (index + 1) / subdivisions]),
    [0, 1]
  ];
  const second = [
    [1, 0],
    [2, 0],
    [2, 1],
    [1, 1],
    ...Array.from({length: subdivisions - 1}, (_, index) => [
      1,
      (subdivisions - index - 1) / subdivisions
    ])
  ];
  const third = [
    [2, 0],
    [3, 0],
    [3, 1],
    [2, 1]
  ];
  return {
    positions: Float32Array.from([...first, ...second, ...third].flat()),
    ringOffsets: Uint32Array.of(
      0,
      first.length,
      first.length + second.length,
      first.length + second.length + third.length
    ),
    polygonOffsets: Uint32Array.of(0, 1, 2, 3),
    featureOffsets: Uint32Array.of(0, 1, 2, 3)
  };
}
