// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The Dixie Fire NDVI scenes as snapshots of a field for the EOF operations. A cell with any missing
// observation in the dates used is dropped by the operations (complete cases), so the field is
// repacked for the first `sliceCount` dates and the CPU reference below applies the same rule.

import {sampleNdviSeries} from '../../../../data/ndvi';
import {ACQUISITION_DATES} from './acquisitions';

export const SNAPSHOT_COUNT = ACQUISITION_DATES.length;

export type SnapshotField = {
  columns: number;
  rows: number;
  cellCount: number;
  /** NDVI, `cell * SNAPSHOT_COUNT + slice`; NaN where the QA mask rejected the observation. */
  observations: Float32Array;
};

export function createSnapshotField(aspect: number, rows: number): SnapshotField {
  const columns = Math.max(8, Math.round(rows * aspect));
  const dates = Array.from({length: SNAPSHOT_COUNT}, (_, index) => index);
  return {
    columns,
    rows,
    cellCount: columns * rows,
    observations: sampleNdviSeries(columns, rows, dates)
  };
}

/** The first `sliceCount` dates, `cell * sliceCount + slice`. */
export function packSnapshots(field: SnapshotField, sliceCount: number): Float32Array {
  const packed = new Float32Array(field.cellCount * sliceCount);
  for (let cell = 0; cell < field.cellCount; cell++) {
    for (let slice = 0; slice < sliceCount; slice++) {
      packed[cell * sliceCount + slice] = field.observations[cell * SNAPSHOT_COUNT + slice];
    }
  }
  return packed;
}

/**
 * The Gram matrix `X' X` of the temporal-mean-removed field over complete cells, in binary64: the
 * matrix that GPURasterSnapshotCovariance reads back, for a run that needs it before the GPU has
 * produced it. Cells with a missing sample or a sample above `maximumValueMagnitude` are dropped.
 */
export function getGramMatrix(
  packed: Float32Array,
  cellCount: number,
  sliceCount: number,
  maximumValueMagnitude: number
): {matrix: Float64Array; keptCellCount: number} {
  const matrix = new Float64Array(sliceCount * sliceCount);
  const anomaly = new Float64Array(sliceCount);
  let keptCellCount = 0;
  for (let cell = 0; cell < cellCount; cell++) {
    let mean = 0;
    let complete = true;
    for (let slice = 0; slice < sliceCount; slice++) {
      const value = packed[cell * sliceCount + slice];
      if (!Number.isFinite(value) || Math.abs(value) > maximumValueMagnitude) {
        complete = false;
        break;
      }
      mean += value / sliceCount;
    }
    if (!complete) continue;
    keptCellCount++;
    for (let slice = 0; slice < sliceCount; slice++)
      anomaly[slice] = packed[cell * sliceCount + slice] - mean;
    for (let row = 0; row < sliceCount; row++) {
      for (let column = 0; column < sliceCount; column++) {
        matrix[row * sliceCount + column] += anomaly[row] * anomaly[column];
      }
    }
  }
  return {matrix, keptCellCount};
}
