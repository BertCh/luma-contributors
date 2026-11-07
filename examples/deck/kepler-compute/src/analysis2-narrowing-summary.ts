// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  RATE_INVALID_CAUSES,
  classifyRateInvalidCauses,
  diagnoseFloat32Narrowing,
  mergeNumericNarrowingReports,
  type NumericNarrowingReport,
  type RateInvalidCause,
  type WideNumericFormat
} from './analysis-numeric-diagnostics';
import {describeNumericNarrowingReport, describeRateCauseCounts} from './analysis2-policy-binding';

/**
 * Wide numeric columns are scanned in full, one Arrow chunk at a time, reading each chunk's typed
 * array directly (no per-row boxing) and merging the chunk reports. A chunk with nulls is read
 * through its validity bitmap instead.
 */
export function scanWideNumericColumn(
  column: arrow.Vector,
  format: WideNumericFormat
): NumericNarrowingReport {
  const reports: NumericNarrowingReport[] = [];
  for (const chunk of column.data) {
    if (chunk.nullCount === 0 && ArrayBuffer.isView(chunk.values)) {
      // The chunk's Float64Array, BigInt64Array or BigUint64Array is scanned in place.
      const values = chunk.values as unknown as {
        subarray(begin: number, end: number): ArrayLike<number | bigint>;
      };
      reports.push(
        diagnoseFloat32Narrowing(format, values.subarray(chunk.offset, chunk.offset + chunk.length))
      );
      continue;
    }
    const vector = new arrow.Vector([chunk]);
    const values: (number | bigint | null)[] = new Array(chunk.length);
    for (let row = 0; row < chunk.length; row++) values[row] = vector.get(row);
    reports.push(diagnoseFloat32Narrowing(format, values));
  }
  return mergeNumericNarrowingReports(reports, format);
}

/**
 * One summary line per Float64 or Int64 column: whether Float32 narrowing is exact, lossy or
 * would overflow, over every row. Informational: the preparation policy decides what actually
 * happens.
 */
export function describeWideNumericNarrowing(table: arrow.Table): string[] {
  const lines: string[] = [];
  for (const field of table.schema.fields) {
    const format: WideNumericFormat | undefined = arrow.DataType.isFloat(field.type)
      ? field.type.precision === arrow.Precision.DOUBLE
        ? 'float64'
        : undefined
      : arrow.DataType.isInt(field.type) && field.type.bitWidth === 64
        ? field.type.isSigned
          ? 'int64'
          : 'uint64'
        : undefined;
    const column = format ? table.getChild(field.name) : null;
    if (!format || !column) continue;
    lines.push(
      `Narrowing ${field.name}: ${describeNumericNarrowingReport(scanWideNumericColumn(column, format))}`
    );
  }
  return lines;
}

/** Cause counts for rate rows (event and population columns) over every row, chunk by chunk. */
export function describeRateSourceExclusions(
  table: arrow.Table,
  eventColumn: string,
  populationColumn: string
): string | undefined {
  const events = table.getChild(eventColumn);
  const populations = table.getChild(populationColumn);
  if (!events || !populations) return undefined;
  const counts = Object.fromEntries(RATE_INVALID_CAUSES.map(cause => [cause, 0])) as Record<
    RateInvalidCause,
    number
  >;
  const blockRows = 65536;
  // Scan fixed row blocks so the temporary number arrays stay small for very large tables.
  for (let start = 0; start < table.numRows; start += blockRows) {
    const rowCount = Math.min(blockRows, table.numRows - start);
    const eventValues: (number | null)[] = new Array(rowCount);
    const populationValues: (number | null)[] = new Array(rowCount);
    for (let row = 0; row < rowCount; row++) {
      const event = events.get(start + row);
      const population = populations.get(start + row);
      eventValues[row] = event === null || event === undefined ? null : Number(event);
      populationValues[row] =
        population === null || population === undefined ? null : Number(population);
    }
    const block = classifyRateInvalidCauses(eventValues, populationValues).counts;
    for (const cause of RATE_INVALID_CAUSES) counts[cause] += block[cause];
  }
  return describeRateCauseCounts(counts);
}
