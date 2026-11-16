// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogOutputTable} from '../../src/analysis/catalog-geojson';
import {
  computeBinPValues,
  computeAdjustedBinPValues,
  computeEmergingHotSpotCells,
  type EmergingHotSpotCubeShape
} from '../../../analysis/test/gpu-spatial-analysis/emerging-hot-spots/emerging-hot-spots-oracle';

/** CPU oracle over independently checked z scores; never derives expectations from returned probabilities. */
export function checkEmergingHotSpotsProbabilitiesAndTrends(
  bins: CatalogOutputTable,
  cells: CatalogOutputTable,
  shape: EmergingHotSpotCubeShape,
  packed: Float32Array
) {
  const mismatches: string[] = [];
  const scores = bins.columns['gi_z'].map(value => (typeof value === 'number' ? value : NaN));
  const raw = computeBinPValues(scores);
  const adjusted = computeAdjustedBinPValues(scores, 'benjamini-hochberg');
  const compare = (
    column: string,
    row: number,
    actual: unknown,
    expected: number,
    absoluteTolerance: number,
    relativeTolerance: number
  ) => {
    if (!Number.isFinite(expected)) {
      if (actual !== null) mismatches.push(`${column} ${row}: ${actual} != null`);
    } else if (
      typeof actual !== 'number' ||
      !Number.isFinite(actual) ||
      Math.abs(actual - expected) > absoluteTolerance + Math.abs(expected) * relativeTolerance
    ) {
      mismatches.push(`${column} ${row}: ${actual} != ${expected}`);
    }
  };
  for (let bin = 0; bin < scores.length; bin++) {
    // Float32 erfcc and sort/scan correction versus double precision: tight enough to reject
    // raw p-values substituted for adjusted p-values, including the committed two fixtures.
    compare('p_value', bin, bins.columns['p_value'][bin], raw[bin], 2e-6, 1e-4);
    compare(
      'adjusted_p_value',
      bin,
      bins.columns['adjusted_p_value'][bin],
      adjusted[bin],
      2e-6,
      1e-4
    );
  }
  const oracle = computeEmergingHotSpotCells(scores, shape, packed, {adjustedPValues: adjusted});
  for (let cell = 0; cell < shape.gridWidth * shape.gridHeight; cell++) {
    compare('trend_z', cell, cells.columns['trend_z'][cell], oracle.trendZ[cell], 2e-3, 1e-3);
    compare('trend_p', cell, cells.columns['trend_p'][cell], oracle.trendP[cell], 2e-6, 1e-4);
    const expectedLastSlice =
      oracle.lastValidSlice[cell] === 0xffffffff ? null : oracle.lastValidSlice[cell];
    if (cells.columns['last_valid_slice'][cell] !== expectedLastSlice)
      mismatches.push(
        `last_valid_slice ${cell}: ${cells.columns['last_valid_slice'][cell]} != ${expectedLastSlice}`
      );
  }
  return {scores, oracle, mismatches};
}
