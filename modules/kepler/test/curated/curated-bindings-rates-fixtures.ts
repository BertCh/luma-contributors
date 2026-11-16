// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';

/** Coincident geometry exceeds GPU topology capacity; all rates remain valid for CPU fallback. */
export function createRateClusterOverflowDataset(): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: Array.from({length: 66}, (_, index) => ({
      id: `coincident-${index}`,
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0]
          ]
        ]
      },
      properties: {reported: index * index, exposed: 5000}
    }))
  };
}

/** Existing class-comparison exclusion band; numerical p-value checks use tighter tolerances. */
const RATE_CLUSTER_CLASS_BOUNDARY_TOLERANCE = 0.01;

type RateClusterClassComparison = {
  sourceIndex: number;
  adjustedPValue: number | null;
  oracleAdjustedPValue: number;
  actualClass: string;
  significantClass: string;
};

/**
 * Compare every published gate and every oracle class outside the boundary band. The number of
 * rows near the threshold is evidence to report, never a device-dependent pass/fail condition.
 */
export function compareRateClusterClasses(
  rows: readonly RateClusterClassComparison[],
  effectiveSignificance: number
) {
  const mismatches: string[] = [];
  const comparedSourceIndices: number[] = [];
  const nearThresholdRows: (RateClusterClassComparison & {oracleClass: string})[] = [];
  let testedCount = 0;
  for (const row of rows) {
    const expectedClass =
      row.adjustedPValue === null
        ? 'not-tested'
        : row.adjustedPValue <= effectiveSignificance
          ? row.significantClass
          : 'not-significant';
    // This gate check is unconditional, including every near-threshold row.
    if (row.actualClass !== expectedClass)
      mismatches.push(`class/gate[${row.sourceIndex}] ${row.actualClass} != ${expectedClass}`);
    if (!Number.isFinite(row.oracleAdjustedPValue)) continue;
    testedCount++;
    const oracleClass =
      row.oracleAdjustedPValue <= effectiveSignificance ? row.significantClass : 'not-significant';
    if (
      Math.abs(row.oracleAdjustedPValue - effectiveSignificance) <=
      RATE_CLUSTER_CLASS_BOUNDARY_TOLERANCE
    ) {
      nearThresholdRows.push({...row, oracleClass});
    } else {
      comparedSourceIndices.push(row.sourceIndex);
      if (row.actualClass !== oracleClass)
        mismatches.push(`oracle class[${row.sourceIndex}] ${row.actualClass} != ${oracleClass}`);
    }
  }
  return {
    testedCount,
    comparedSourceIndices,
    nearThresholdRows,
    mismatches,
    effectiveSignificance,
    boundaryTolerance: RATE_CLUSTER_CLASS_BOUNDARY_TOLERANCE
  };
}
