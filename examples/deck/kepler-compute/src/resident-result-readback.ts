// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  prepareArrowTableForGPUAnalytics,
  type PreparedArrowAnalyticsTable
} from '@luma-contributors/analysis/analysis-core';
import type {KeplerComputeSession} from '@luma-contributors/kepler';
import * as arrow from 'apache-arrow';

/**
 * Results with at most this many rows are read back, validated on the CPU and adopted with
 * `adoptArrowDataset` (full cells, device recovery, every export path). Larger results stay
 * GPU-resident and lazy (`adoptResidentDataset`). Matches the native Jenks and cluster cap.
 */
export const MAXIMUM_RESIDENT_READBACK_ROW_COUNT = 262144;

export function shouldReadBackResidentResult(rowCount: number): boolean {
  return rowCount <= MAXIMUM_RESIDENT_READBACK_ROW_COUNT;
}

export type ResidentReadbackFormat = 'float32' | 'sint32' | 'uint32';

export type ResidentReadbackColumn = {
  name: string;
  format: ResidentReadbackFormat;
  /** Labels for uint32 category codes; valid rows must reference an existing label. */
  dictionary?: readonly string[];
  /** Refuse valid float32 cells that are NaN or infinite. */
  refuseNonFinite?: boolean;
  /** One entry per GPU chunk, in row order. */
  chunks: readonly {values: Uint32Array; validity: Uint32Array}[];
};

export type ResidentReadbackTable = {
  table: arrow.Table;
  prepared: PreparedArrowAnalyticsTable;
  normalization: {dictionaries: Record<string, {values: readonly string[]; ordered: boolean}>};
};

/** CPU memory bound for readback words, Arrow copies, string rows and dictionary preparation. */
export function estimateResidentReadbackCpuByteLength(
  columns: readonly {dictionary?: readonly string[]}[],
  rowCount: number
): number {
  let labelBytes = 0;
  for (const column of columns)
    for (const label of column.dictionary ?? []) labelBytes += label.length * 2;
  return rowCount * columns.length * 40 + labelBytes * 4 + 65536;
}

/**
 * Working-memory ceiling for the GPU min/max domain reduction over a resident result. Each domain
 * owns validity masks, finite-value query outputs and reduction scratch (measured about 21 bytes
 * per row and column plus selection storage), so the ceiling scales with the column count. The
 * reservation is resized to the measured usage once the explorer is built.
 */
export function getResidentDomainBudgetBytes(rowCount: number, columnCount: number): number {
  return 8 * 1024 * 1024 + rowCount * (64 + 32 * columnCount);
}

/** Whether a float32 bit pattern is NaN or infinite (all exponent bits set). */
export function isNonFiniteFloatBits(bits: number): boolean {
  return (bits & 0x7f800000) === 0x7f800000;
}

/**
 * Validates GPU words and builds the retained Arrow table with the GPU chunk layout as batches.
 * Refuses validity other than 0 or 1, dictionary codes outside the labels on valid rows, and
 * non-finite cells where requested. Invalid rows become Arrow nulls.
 */
export function makeResidentReadbackTable(props: {
  columns: readonly ResidentReadbackColumn[];
  metadata: Readonly<Record<string, string>>;
}): ResidentReadbackTable {
  const {columns} = props;
  const chunkCount = columns[0]?.chunks.length ?? 0;
  const normalization: ResidentReadbackTable['normalization'] = {dictionaries: {}};
  const batches: arrow.RecordBatch[] = [];
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    const vectors: Record<string, arrow.Vector> = Object.create(null);
    for (const column of columns) {
      const {values, validity} = column.chunks[chunk];
      if (values.length !== validity.length)
        throw new Error('Result validity does not match its values');
      const labels = column.dictionary;
      const floats =
        column.format === 'float32'
          ? new Float32Array(values.buffer, values.byteOffset, values.length)
          : undefined;
      const integers =
        column.format === 'sint32'
          ? new Int32Array(values.buffer, values.byteOffset, values.length)
          : undefined;
      const rows: (number | string | null)[] = new Array(values.length);
      for (let row = 0; row < values.length; row++) {
        if (validity[row] > 1) throw new Error('Result validity must be 0 or 1');
        if (!validity[row]) {
          rows[row] = null;
        } else if (labels) {
          if (values[row] >= labels.length)
            throw new Error('Result has an invalid dictionary code');
          rows[row] = labels[values[row]];
        } else if (floats) {
          if (column.refuseNonFinite && isNonFiniteFloatBits(values[row]))
            throw new Error('Result has a non-finite value; use SQL for unsupported values');
          rows[row] = floats[row];
        } else {
          rows[row] = integers ? integers[row] : values[row];
        }
      }
      vectors[column.name] = arrow.vectorFromArray(
        rows,
        labels
          ? new arrow.Utf8()
          : column.format === 'float32'
            ? new arrow.Float32()
            : column.format === 'sint32'
              ? new arrow.Int32()
              : new arrow.Uint32()
      );
      if (labels) normalization.dictionaries[column.name] = {values: labels, ordered: false};
    }
    batches.push(...new arrow.Table(vectors).batches);
  }
  const table = batches.length ? new arrow.Table(batches) : new arrow.Table();
  for (const [key, value] of Object.entries(props.metadata)) table.schema.metadata.set(key, value);
  const prepared = prepareArrowTableForGPUAnalytics(table, normalization);
  // Seeded preparation must preserve the exact GPU codes of valid rows.
  for (const [columnIndex, column] of columns.entries()) {
    if (!column.dictionary) continue;
    const preparedColumn = prepared.table.getChildAt(columnIndex)!;
    let rowOffset = 0;
    for (const chunk of column.chunks) {
      for (let row = 0; row < chunk.values.length; row++)
        if (
          chunk.validity[row] &&
          getPreparedCode(preparedColumn, rowOffset + row) !== chunk.values[row]
        )
          throw new Error('Prepared result dictionary changed adopted GPU codes');
      rowOffset += chunk.values.length;
    }
  }
  return {table, prepared, normalization};
}

function getPreparedCode(vector: arrow.Vector, row: number): number | undefined {
  let offset = row;
  for (const data of vector.data) {
    if (offset < data.length) return data.values[data.offset + offset];
    offset -= data.length;
  }
  return undefined;
}

/**
 * Reads every chunk of the given GPU vectors, one column at a time so at most two staging buffers
 * exist at once. Returns the words and the number of bytes read.
 */
export async function readResidentResultChunks(props: {
  session: KeplerComputeSession;
  columns: readonly {
    name: string;
    format: ResidentReadbackFormat;
    dictionary?: readonly string[];
    refuseNonFinite?: boolean;
    values: GPUVector;
    validity: GPUVector<'uint32'>;
  }[];
  assertCurrent(): void;
}): Promise<{columns: ResidentReadbackColumn[]; readbackBytes: number}> {
  const columns: ResidentReadbackColumn[] = [];
  let readbackBytes = 0;
  for (const column of props.columns) {
    const chunks: ResidentReadbackColumn['chunks'][number][] = [];
    if (column.values.data.length !== column.validity.data.length)
      throw new Error('Result validity does not match its values');
    for (const [index, data] of column.values.data.entries()) {
      const validityData = column.validity.data[index];
      if (data.length !== validityData.length)
        throw new Error('Result validity does not match its values');
      if (!data.length) continue;
      const [valueBytes, validityBytes] = await Promise.all([
        Promise.resolve().then(() =>
          (data.buffer as Buffer).readAsync(data.byteOffset, data.length * 4)
        ),
        Promise.resolve().then(() =>
          (validityData.buffer as Buffer).readAsync(validityData.byteOffset, data.length * 4)
        )
      ]);
      readbackBytes += valueBytes.byteLength + validityBytes.byteLength;
      props.assertCurrent();
      chunks.push({
        values: new Uint32Array(valueBytes.slice().buffer),
        validity: new Uint32Array(validityBytes.slice().buffer)
      });
    }
    columns.push({
      name: column.name,
      format: column.format,
      dictionary: column.dictionary,
      refuseNonFinite: column.refuseNonFinite,
      chunks
    });
  }
  return {columns, readbackBytes};
}
