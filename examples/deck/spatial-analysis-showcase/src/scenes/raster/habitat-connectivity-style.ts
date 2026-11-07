// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** ESA WorldCover class codes in stable categorical raster order. */
export const WORLD_COVER_CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100] as const;

/** Labels and colours share the raw-class lookup used to build the displayed raster. */
export const WORLD_COVER_CLASSES = [
  {code: 10, label: 'Tree cover', color: [0, 100, 0, 235]},
  {code: 20, label: 'Shrubland', color: [110, 145, 45, 235]},
  {code: 30, label: 'Grassland', color: [220, 205, 55, 235]},
  {code: 40, label: 'Cropland', color: [205, 135, 205, 235]},
  {code: 50, label: 'Built-up', color: [210, 65, 55, 235]},
  {code: 60, label: 'Bare / sparse vegetation', color: [165, 165, 160, 235]},
  {code: 70, label: 'Snow and ice', color: [225, 230, 230, 235]},
  {code: 80, label: 'Permanent water', color: [40, 110, 180, 235]},
  {code: 90, label: 'Herbaceous wetland', color: [30, 150, 135, 235]},
  {code: 95, label: 'Mangroves', color: [120, 80, 145, 235]},
  {code: 100, label: 'Moss and lichen', color: [100, 125, 80, 235]}
] as const;

/** Stable patch-label colours; labels above the palette length cycle. */
export const HABITAT_PATCH_COLORS: readonly (readonly [number, number, number, number])[] = [
  [0, 114, 178, 235],
  [230, 159, 0, 235],
  [0, 158, 115, 235],
  [204, 121, 167, 235],
  [86, 180, 233, 235],
  [213, 94, 0, 235]
];
