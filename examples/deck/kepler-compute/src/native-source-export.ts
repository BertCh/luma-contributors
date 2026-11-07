// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {timeToUnixMilli} from './native-time-converter';
import type {KeplerTable} from '@kepler.gl/table';
import {makeArrowColumnFromNativeField} from './native-column-arrow';
import {BoundedByteCollector, formatCSVValue, quoteCSV} from './source-export';

export type NativeExportFormat = 'arrow' | 'csv';
export type NativeExportSelection = 'filtered' | 'all';

export type NativeExportPlan = {
  dataset: Pick<KeplerTable, 'id' | 'label' | 'fields' | 'dataContainer' | 'dataRevision'> & {
    filteredIndex?: ArrayLike<number> | null;
  };
  /** Native row indices to write, in native order. */
  rowIndices: Uint32Array;
  /** Stable Arrow type per field so every page of a streamed export shares one schema. */
  schema: arrow.Schema;
  integerRoutes: Readonly<Record<string, 'int32' | 'uint32' | 'float64' | 'utf8'>>;
};

export type NativeExportResult = {
  bytes: Uint8Array<ArrayBuffer>;
  format: NativeExportFormat;
  rowCount: number;
  datasetId: string;
  datasetRevision: number;
};

export type NativeExportDataset = NativeExportPlan['dataset'];

/** Rows to export: the native filter result when requested and present, otherwise every row. */
export function selectNativeRows(
  dataset: NativeExportDataset,
  selection: NativeExportSelection
): Uint32Array {
  const rowCount = dataset.dataContainer.numRows();
  const filtered = dataset.filteredIndex;
  if (selection === 'filtered' && filtered) return Uint32Array.from(filtered);
  return Uint32Array.from({length: rowCount}, (_value, index) => index);
}

/** One scan of integer fields fixes their Arrow type; other types follow the Kepler field type. */
export function planNativeExport(
  dataset: NativeExportDataset,
  selection: NativeExportSelection
): NativeExportPlan {
  const rowIndices = selectNativeRows(dataset, selection);
  const integerRoutes: Record<string, 'int32' | 'uint32' | 'float64' | 'utf8'> =
    Object.create(null);
  const fields: arrow.Field[] = [];
  for (const [columnIndex, field] of dataset.fields.entries()) {
    if (field.type === 'integer') {
      let isInt32 = true;
      let isUint32 = true;
      let isIntegral = true;
      for (const rowIndex of rowIndices) {
        const raw = dataset.dataContainer.valueAt(rowIndex, columnIndex);
        const value = raw === null || raw === undefined || raw === '' ? null : Number(raw);
        if (value === null || Number.isNaN(value)) continue;
        if (!Number.isInteger(value)) isIntegral = false;
        if (value < -(2 ** 31) || value > 2 ** 31 - 1) isInt32 = false;
        if (value < 0 || value > 2 ** 32 - 1) isUint32 = false;
      }
      integerRoutes[field.name] =
        isIntegral && isInt32
          ? 'int32'
          : isIntegral && isUint32
            ? 'uint32'
            : isIntegral
              ? 'utf8'
              : 'float64';
    }
    fields.push(new arrow.Field(field.name, getNativeArrowType(field, integerRoutes), true));
  }
  return {dataset, rowIndices, schema: new arrow.Schema(fields), integerRoutes};
}

function getNativeArrowType(
  field: {name: string; type: string},
  integerRoutes: Readonly<Record<string, string>>
): arrow.DataType {
  switch (field.type) {
    case 'timestamp':
    case 'date':
      return new arrow.TimestampMillisecond();
    case 'boolean':
      return new arrow.Bool();
    case 'real':
      return new arrow.Float64();
    case 'integer':
      switch (integerRoutes[field.name]) {
        case 'int32':
          return new arrow.Int32();
        case 'uint32':
          return new arrow.Uint32();
        case 'float64':
          return new arrow.Float64();
        default:
          return new arrow.Utf8();
      }
    default:
      return new arrow.Utf8();
  }
}

/** Builds one bounded page; nulls (missing, NaN, empty text) remain Arrow nulls. */
export function makeNativeExportPage(
  plan: NativeExportPlan,
  offset: number,
  limit: number
): arrow.Table {
  const {dataset, rowIndices} = plan;
  const pageRows = rowIndices.subarray(offset, Math.min(rowIndices.length, offset + limit));
  const columns: Record<string, arrow.Vector> = Object.create(null);
  for (const [columnIndex, field] of dataset.fields.entries()) {
    const values = Array.from(pageRows, (rowIndex): unknown =>
      dataset.dataContainer.valueAt(rowIndex, columnIndex)
    );
    const route = plan.integerRoutes[field.name];
    if (field.type === 'integer') {
      const numbers = values.map(value =>
        value === null || value === undefined || value === '' || Number.isNaN(Number(value))
          ? null
          : Number(value)
      );
      columns[field.name] =
        route === 'utf8'
          ? arrow.vectorFromArray(
              numbers.map(number => (number === null ? null : BigInt(number).toString())),
              new arrow.Utf8()
            )
          : arrow.vectorFromArray(
              numbers,
              route === 'int32'
                ? new arrow.Int32()
                : route === 'uint32'
                  ? new arrow.Uint32()
                  : new arrow.Float64()
            );
    } else
      columns[field.name] = makeArrowColumnFromNativeField(
        {name: field.name, type: field.type, format: field.format ?? ''},
        values,
        timeToUnixMilli
      ).vector;
  }
  return new arrow.Table(columns);
}

/** Formats one Arrow page as CSV lines with the same null and timestamp rules as source export. */
export function formatTableCsvRows(table: arrow.Table): string {
  const lines: string[] = [];
  for (let row = 0; row < table.numRows; row++)
    lines.push(
      table.schema.fields
        .map((field, column) =>
          quoteCSV(formatCSVValue(table.getChildAt(column)!, row, field.type))
        )
        .join(',')
    );
  return lines.length ? `${lines.join('\r\n')}\r\n` : '';
}

export function formatCsvHeader(schema: arrow.Schema): string {
  return `${schema.fields.map(field => quoteCSV(field.name)).join(',')}\r\n`;
}

/**
 * Non-streaming export of a native table. It keeps the explicit row and byte limits of
 * `exportSelectedSource`; larger tables use the streaming path.
 */
export async function exportNativeDataset(
  dataset: NativeExportDataset,
  options: {
    format: NativeExportFormat;
    selection: NativeExportSelection;
    maximumRows: number;
    maximumOutputBytes: number;
    signal?: AbortSignal;
  }
): Promise<NativeExportResult> {
  options.signal?.throwIfAborted();
  const plan = planNativeExport(dataset, options.selection);
  if (plan.rowIndices.length > options.maximumRows)
    throw new Error(
      `Native export has ${plan.rowIndices.length.toLocaleString()} rows; the non-streaming limit is ${options.maximumRows.toLocaleString()}. Use Stream download.`
    );
  const revision = dataset.dataRevision;
  const table = makeNativeExportPage(plan, 0, plan.rowIndices.length);
  options.signal?.throwIfAborted();
  if (dataset.dataRevision !== revision) throw new Error('Native dataset changed during export');
  const collector = new BoundedByteCollector(options.maximumOutputBytes);
  if (options.format === 'arrow') {
    const writer = new arrow.RecordBatchStreamWriter();
    writer.reset(collector, plan.schema);
    for (const batch of table.batches) writer.write(new arrow.RecordBatch(plan.schema, batch.data));
    writer.close();
  } else {
    const encoder = new TextEncoder();
    collector.write(encoder.encode(formatCsvHeader(plan.schema)));
    collector.write(encoder.encode(formatTableCsvRows(table)));
  }
  return {
    bytes: collector.finishBytes(),
    format: options.format,
    rowCount: plan.rowIndices.length,
    datasetId: dataset.id,
    datasetRevision: revision
  };
}
