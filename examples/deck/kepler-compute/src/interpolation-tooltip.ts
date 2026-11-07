// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentTooltipValue} from './resident-tooltip-fields';

type TooltipGrid = {
  bounds: readonly [number, number, number, number];
  width: number;
  height: number;
};

/**
 * North-up grid cell containing a longitude/latitude, or null outside the grid. Matches the cell
 * layout of InterpolationRuntime.readPage (row zero at the north edge).
 */
export function getInterpolationCellIndex(
  grid: TooltipGrid,
  longitude: number,
  latitude: number
): number | null {
  const [west, south, east, north] = grid.bounds;
  if (!(longitude >= west && longitude <= east && latitude >= south && latitude <= north))
    return null;
  const column = Math.min(
    grid.width - 1,
    Math.floor(((longitude - west) / (east - west)) * grid.width)
  );
  const row = Math.min(
    grid.height - 1,
    Math.floor(((north - latitude) / (north - south)) * grid.height)
  );
  return row * grid.width + column;
}

export type InterpolationTooltipCell = {
  value: number | null;
  neighbors?: number;
  variance?: number | null;
  lower?: number | null;
  upper?: number | null;
  confidenceLevel?: number;
};

/** Tooltip entries for the interpolated surface value at a picked point's location. */
export function makeInterpolationTooltipValues(
  cell: InterpolationTooltipCell,
  column: string
): Record<string, ResidentTooltipValue> {
  const values: Record<string, ResidentTooltipValue> = {
    [`Interpolated ${column}`]: cell.value
  };
  if (cell.neighbors !== undefined) values['Interpolation neighbors'] = cell.neighbors;
  if (cell.variance !== undefined) values['Kriging variance'] = cell.variance;
  if (cell.lower !== undefined && cell.upper !== undefined) {
    const level = cell.confidenceLevel === undefined ? '' : ` (${cell.confidenceLevel})`;
    values[`Kriging interval lower${level}`] = cell.lower;
    values[`Kriging interval upper${level}`] = cell.upper;
  }
  return values;
}
