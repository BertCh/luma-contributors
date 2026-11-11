// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** The three ways a point-cloud raster grid names its row direction; the default is `'south'`. */
export const POINT_CLOUD_ROW_DIRECTIONS = [
  {label: 'north', rowDirection: 'north' as const},
  {label: 'south', rowDirection: 'south' as const},
  {label: 'default (south)', rowDirection: undefined}
] as const;

export type PointCloudRowDirectionCase = (typeof POINT_CLOUD_ROW_DIRECTIONS)[number];

/**
 * Lays out a raster written with row 0 at the minimum y (north order) for a grid of the given
 * direction: `'north'` keeps it, `'south'` and the default reverse the rows, so the same cell holds
 * the same value in every direction.
 */
export function arrangeRasterRows<Value>(
  northValues: readonly Value[],
  width: number,
  height: number,
  rowDirection: 'north' | 'south' | undefined
): Value[] {
  if (rowDirection === 'north') {
    return [...northValues];
  }
  const rows: Value[] = [];
  for (let row = height - 1; row >= 0; row--) {
    rows.push(...northValues.slice(row * width, (row + 1) * width));
  }
  return rows;
}

/** Spreads a grid with the direction under test; the default case leaves `rowDirection` out. */
export function withPointCloudRowDirection<Grid extends object>(
  grid: Grid,
  rowDirection: 'north' | 'south' | undefined
): Grid & {rowDirection?: 'north' | 'south'} {
  return rowDirection ? {...grid, rowDirection} : grid;
}
