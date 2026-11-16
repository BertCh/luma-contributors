// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogRunOutput} from '../../src/analysis/catalog-binding-types';
import {
  makeCatalogRasterInput,
  type CatalogRasterBand
} from '../../src/analysis/catalog-raster-input';

/** Two raw values collide in float32; only the exact source-domain sentinel is missing. */
export function createDrainageNoDataCollisionRaster(
  values: CatalogRasterBand['values'],
  noDataValue: number
) {
  return makeCatalogRasterInput(
    {
      id: 'drainage-nodata-collision',
      crs: 'planar',
      cellSizeMode: 'uniform',
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      band: {index: 1, values, validity: new Uint8Array([1, 1]), noDataValue}
    },
    2
  );
}

/** Include numerical evidence in errors; the headless runner does not retain console logs. */
export function compareDrainageColumn(
  result: CatalogRunOutput,
  column: string,
  expected: readonly number[],
  tolerance = 0
): void {
  const actual = result.tables.find(table => table.id === 'catchment-cells')!.columns[column];
  if (
    actual.length !== expected.length ||
    expected.some(
      (value, row) =>
        typeof actual[row] !== 'number' ||
        !Number.isFinite(actual[row]) ||
        !Number.isFinite(value) ||
        Math.abs(Number(actual[row]) - value) > tolerance
    )
  ) {
    const report = JSON.stringify(
      {actual, expected, tolerance, summary: result.summary, diagnostics: result.diagnostics},
      (_key, value) =>
        typeof value === 'number' && !Number.isFinite(value) ? String(value) : value
    );
    throw new Error(`drainage ${column} mismatch: ${report}`);
  }
}
