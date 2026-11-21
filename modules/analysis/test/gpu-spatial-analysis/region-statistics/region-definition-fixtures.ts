// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** R14 composition inputs shared by the headless case and emitted cell-range probe.
 * Rows occupy columns 1 and 3, so the recovery query plus its one-cell halo excludes row 1.
 */
export const REGION_MASK_COMPOSITION_GRID = {
  width: 4,
  bounds: [0, 0, 1, 1] as const,
  positions: Float32Array.of(0.25, 0.25, 0.75, 0.25),
  initialQuery: Float32Array.of(-1, -1, 2, 2),
  completeQuery: Float32Array.of(0, 0, 0.49, 1),
  cellOffsets: Uint32Array.of(0, 0, 1, 1, 2)
};

export const DEFINITION_POSITIONS = Float32Array.of(
  0,
  0,
  2,
  0,
  2,
  2,
  0,
  2,
  1,
  1,
  3,
  1,
  -1,
  1,
  1,
  2,
  1,
  -1,
  2,
  1,
  4,
  4,
  1,
  0
);
export const DEFINITION_VALUES = Float32Array.of(
  4,
  8,
  8,
  Number.NaN,
  -2,
  20,
  Number.POSITIVE_INFINITY,
  1,
  6,
  0,
  10,
  2
);
export const DEFINITION_IDS = Uint32Array.of(91, 7, 43, 500, 12, 42, 19, 3, 80, 77, 9, 101);

// NumPy/Shapely literals from region-definition.oracle.py. The summary's float
// words are IEEE-754 encodings, generated independently of the decoder.
export const REGION_DEFINITION = {
  square: {
    words: [8, 7, 1101529088, 1077936128, 3221225472, 1090519040, 1, 1, 2, 1, 1, 2],
    mask: [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    ids: [91, 7, 43, 500, 12],
    count: 5,
    requiredCount: 8,
    overflow: 1,
    sum: 21,
    mean: 3,
    minimum: -2,
    maximum: 8
  },
  radius: {
    words: [4, 4, 1065353216, 1048576000, 3221225472, 1073741824, 1, 0, 2, 1, 0, 0],
    mask: [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
    ids: [12, 3, 77, 101],
    count: 4,
    requiredCount: 4,
    overflow: 0,
    sum: 1,
    mean: 0.25,
    minimum: -2,
    maximum: 2
  },
  empty: {
    words: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    mask: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ids: [],
    count: 0,
    requiredCount: 0,
    overflow: 0,
    sum: 0,
    mean: 0,
    minimum: 0,
    maximum: 0
  }
} as const;
