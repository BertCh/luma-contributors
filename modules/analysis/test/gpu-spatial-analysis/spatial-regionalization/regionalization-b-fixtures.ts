// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// NumPy literals from regionalization-b.oracle.py: two connected four-row graphs and an island.
export const AUDIT_VALUES = [0, 2, 10, 14, 1, 3, 20, 26, 99];
export const AUDIT_OFFSETS = [0, 2, 5, 8, 10, 12, 15, 18, 20, 20];
export const AUDIT_NEIGHBORS = [1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 5, 6, 4, 6, 7, 4, 5, 7, 5, 6];
export const AUDIT_STANDARDIZED = [
  -0.6611639736922161, -0.5931585363981596, -0.3211367872219335, -0.18512591263382047,
  -0.6271612550451878, -0.5591558177511313, 0.018890399248349087, 0.22290671113051866,
  2.7051051723635817
];
export const AUDIT_FOREST_SLOTS = [0, 3, 7, 10, 13, 17];
export const AUDIT_FOREST_COSTS = [
  0.004624739501555851, 0.07399583202489368, 0.018498958006223414, 0.004624739501555851,
  0.33413742898741045, 0.04162265551400268
];
export const AUDIT_WARD_PREFIXES = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8],
  [0, 1, 2, 3, 4, 4, 6, 7, 8],
  [0, 0, 2, 3, 4, 4, 6, 7, 8],
  [0, 0, 2, 2, 4, 4, 6, 7, 8],
  [0, 0, 2, 2, 4, 4, 6, 6, 8],
  [0, 0, 0, 0, 4, 4, 6, 6, 8],
  [0, 0, 0, 0, 4, 4, 4, 4, 8]
];
export const AUDIT_WARD_OBJECTIVES = [0, 2, 4, 12, 30, 151, 592];

export function createRegionalizationAuditInput(scale = 1, shift = 0) {
  return {
    offsets: Uint32Array.from(AUDIT_OFFSETS),
    neighbors: Uint32Array.from(AUDIT_NEIGHBORS),
    values: Float32Array.from(AUDIT_VALUES, value => (value + shift) * scale),
    parameters: Uint32Array.of(0)
  };
}
