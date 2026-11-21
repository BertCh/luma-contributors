// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** All supported nonempty subsets; no contributor accepts an empty output object. */
export const CLASSIFICATION_OUTPUT_SUBSETS = [
  {colors: true, classes: false, counts: false},
  {colors: false, classes: true, counts: false},
  {colors: false, classes: false, counts: true},
  {colors: true, classes: true, counts: false},
  {colors: true, classes: false, counts: true},
  {colors: false, classes: true, counts: true},
  {colors: true, classes: true, counts: true}
] as const;

// Hand derivation: threshold edges 2 and 5 put -1 below both, 2, 4 above one,
// and 5, 9, 8 above both. NaN and masked row 7 are unusable. Capacity 4 has a zero tail.
// These integer outputs need no float32 tolerance; all comparisons are exact.
export const COLOR_OPTIONAL_FIXTURE = {
  values: [-1, 2, 4, 5, 9, 8, NaN, 3],
  mask: [1, 1, 1, 1, 1, 1, 1, 0],
  domain: [-10, 2, 5, 10],
  palette: [0xff000011, 0xff000022, 0xff000033, 0xff000044],
  classes: [0, 1, 1, 2, 2, 2, 0xffffffff, 0xffffffff],
  colors: [0xff000011, 0xff000022, 0xff000022, 0xff000033, 0xff000033, 0xff000033, 7, 7],
  counts: [1, 2, 3, 0]
};

// X edge 2, Y edge 5; class = 2 * (y >= 5) + (x >= 2). This distinguishes
// swapped axes even with only counts requested: class 1 has two rows, class 2 only one.
// Mask/NaN remove the last two rows. Capacity 3 has 5 zero tails.
// Palette alpha is 128, so alpha factors 1/2, 1, 3/4 produce exact bytes 64, 128, 96.
export const BIVARIATE_OPTIONAL_FIXTURE = {
  valuesX: [0, 2, 0, 2, 3, 2, NaN, 0],
  valuesY: [0, 0, 5, 5, 9, 0, 0, 0],
  mask: [1, 1, 1, 1, 1, 1, 1, 0],
  alphaValues: [0, 1, 0.5, NaN, 1, 0.5, 1, 1],
  breaksX: [-10, 2, 10],
  breaksY: [-10, 5, 10],
  palette: [0x80000011, 0x80000022, 0x80000033, 0x80000044],
  classes: [0, 1, 2, 3, 3, 1, 0xffffffff, 0xffffffff],
  colors: [0x80000011, 0x80000022, 0x80000033, 0x80000044, 0x80000044, 0x80000022, 7, 7],
  fadedColors: [0x40000011, 0x80000022, 0x60000033, 0x40000044, 0x80000044, 0x60000022, 7, 7],
  counts: [1, 2, 1, 2, 0, 0, 0, 0, 0]
};
