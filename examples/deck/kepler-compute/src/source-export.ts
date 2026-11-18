// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {formatTimestampTicks} from '@luma-contributors/analysis/dataframe-analysis';
import {readOriginalTimestampTicks, resolveOriginalColumnRow} from './original-column-values';
import {
  materializeKeplerSelection,
  type KeplerComputeSession,
  type KeplerSessionExplorer,
  type KeplerSelectionExportOptions
} from '@luma-contributors/kepler';

import {materializeResidentRows} from './resident-materialization';

export type SourceExportFormat = 'arrow' | 'csv';
export type SourceExportMetadata = {
  datasetId: string;
  datasetRevision: number;
  selectionRevision: number;
  selectedCount: number;
  offset: number;
};
export type SerializedSourceExport = {
  bytes: Uint8Array<ArrayBuffer>;
  format: SourceExportFormat;
  rowCount: number;
  sourceRowColumn: string;
  recordBatchCount: number;
  derivedColumns?: Record<string, string>;
};

/** Explicit export of original retained source values selected by GPU row identities. */
export async function exportSelectedSource(
  session: KeplerComputeSession,
  explorer: KeplerSessionExplorer,
  options: (
    | Omit<Extract<KeplerSelectionExportOptions, {mode: 'page'}>, 'columns'>
    | Omit<Extract<KeplerSelectionExportOptions, {mode: 'all-selected'}>, 'columns'>
  ) & {
    format: SourceExportFormat;
    maximumOutputBytes: number;
  }
): Promise<
  SerializedSourceExport & SourceExportMetadata & {readbackBytes: number; destroy(): void}
> {
  validateBudget(options.maximumOutputBytes);
  const dataset = explorer.dataset;
  const indices = await materializeKeplerSelection(session, explorer, {...options, columns: []});
  let reservation: ReturnType<typeof session.registry.reserveMemory> | undefined;
  const metadata: SourceExportMetadata = {
    datasetId: indices.datasetId,
    datasetRevision: indices.datasetRevision,
    selectionRevision: indices.selectionRevision,
    selectedCount: indices.selectedCount,
    offset: indices.offset
  };
  const assertCurrent = () => {
    options.signal?.throwIfAborted();
    if (
      !explorer.isCurrent ||
      explorer.currentRevision !== indices.selectionRevision ||
      session.registry.getDataset(dataset.id) !== dataset
    )
      throw new Error('Source export dataset or selection is stale');
  };
  try {
    // IPC/CSV chunks and the final contiguous download buffer can coexist. JS metadata is additional.
    reservation = session.registry.reserveMemory(options.maximumOutputBytes * 2, 'cpu', 'output');
    // GPU-resident results page only the selected rows into their accounted lazy source view.
    await materializeResidentRows(dataset.value, indices.sourceRowIndices, {assertCurrent});
    const serialized = await serializeSelectedSource(
      dataset.value.source,
      indices.sourceRowIndices,
      metadata,
      {
        format: options.format,
        maximumOutputBytes: options.maximumOutputBytes,
        assertCurrent
      }
    );
    assertCurrent();
    let destroyed = false;
    const result = {
      ...serialized,
      ...metadata,
      readbackBytes: indices.readbackBytes,
      destroy() {
        if (destroyed) return;
        destroyed = true;
        result.bytes = new Uint8Array();
        reservation?.release();
      }
    };
    return result;
  } catch (error) {
    reservation?.release();
    throw error;
  } finally {
    indices.destroy();
  }
}

/** Arrow preserves exact storage by slicing contiguous source runs; CSV is a text representation. */
export async function serializeSelectedSource(
  source: arrow.Table,
  sourceRows: Uint32Array,
  metadata: SourceExportMetadata,
  options: {
    format: SourceExportFormat;
    maximumOutputBytes: number;
    assertCurrent(): void;
    /** Compact selected-row-aligned scalar results; source storage remains borrowed and unchanged. */
    derived?: {table: arrow.Table; metadata: Readonly<Record<string, unknown>>};
  }
): Promise<SerializedSourceExport> {
  validateBudget(options.maximumOutputBytes);
  options.assertCurrent();
  for (let index = 0; index < sourceRows.length; index++)
    if (
      sourceRows[index] >= source.numRows ||
      (index > 0 && sourceRows[index] <= sourceRows[index - 1])
    )
      throw new Error('Source export requires increasing unique source-row identities');
  const derived = options.derived;
  if (derived && (derived.table.numRows !== sourceRows.length || derived.table.batches.length > 1))
    throw new Error('Derived export requires one compact batch aligned to selected rows');
  const usedNames = new Set(source.schema.fields.map(field => field.name));
  const derivedFields = (derived?.table.schema.fields ?? []).map(field => {
    let name = field.name;
    while (usedNames.has(name)) name += '_analysis';
    usedNames.add(name);
    return new arrow.Field(name, field.type, field.nullable, field.metadata);
  });
  const derivedColumns = derived
    ? Object.fromEntries(
        derived.table.schema.fields.map((field, index) => [field.name, derivedFields[index].name])
      )
    : undefined;
  let sourceRowColumn = '__luma_source_row_index';
  while (usedNames.has(sourceRowColumn)) sourceRowColumn += '_';
  const collector = new BoundedByteCollector(options.maximumOutputBytes);
  let recordBatchCount = 0;
  if (options.format === 'arrow') {
    const fields = [
      ...source.schema.fields,
      ...derivedFields,
      new arrow.Field(sourceRowColumn, new arrow.Uint32(), false)
    ];
    const schemaMetadata = new Map(source.schema.metadata);
    schemaMetadata.set(
      'luma:source-export',
      JSON.stringify({
        ...metadata,
        sourceRowColumn,
        representation: derived ? 'original-source-and-derived-analysis' : 'original-arrow-source',
        ...(derived ? {derivedColumns} : {})
      })
    );
    if (derived)
      schemaMetadata.set(
        'luma:analysis-export',
        JSON.stringify({...derived.metadata, derivedColumns, sourceRowColumn})
      );
    const schema = new arrow.Schema(fields, schemaMetadata);
    const writer = new arrow.RecordBatchStreamWriter();
    writer.reset(collector, schema);
    for (let index = 0; index < sourceRows.length; ) {
      options.assertCurrent();
      let end = index + 1;
      while (end < sourceRows.length && sourceRows[end] === sourceRows[end - 1] + 1) end++;
      let cursor = index;
      for (const batch of source.slice(sourceRows[index], sourceRows[end - 1] + 1).batches) {
        const identities = arrow.makeVector(sourceRows.subarray(cursor, cursor + batch.numRows));
        const data = batch.data.clone(
          new arrow.Struct(fields),
          batch.data.offset,
          batch.numRows,
          batch.data.nullCount,
          batch.data.buffers,
          [
            ...batch.data.children,
            ...derivedFields.map(
              (_field, columnIndex) =>
                derived!.table.getChildAt(columnIndex)!.slice(cursor, cursor + batch.numRows)
                  .data[0]
            ),
            identities.data[0]
          ]
        );
        writer.write(new arrow.RecordBatch(schema, data));
        cursor += batch.numRows;
        recordBatchCount++;
      }
      index = end;
      if (recordBatchCount % 128 === 0) await yieldExport(options.assertCurrent);
    }
    writer.close();
  } else {
    const encoder = new TextEncoder();
    collector.write(
      encoder.encode(
        [
          ...source.schema.fields.map(field => quoteCSV(field.name)),
          ...derivedFields.map(field => quoteCSV(field.name)),
          quoteCSV(sourceRowColumn)
        ].join(',') + '\r\n'
      )
    );
    let batchIndex = 0;
    let batchOffset = 0;
    for (let index = 0; index < sourceRows.length; index++) {
      options.assertCurrent();
      const row = sourceRows[index];
      while (
        batchIndex < source.batches.length &&
        row >= batchOffset + source.batches[batchIndex].numRows
      ) {
        batchOffset += source.batches[batchIndex].numRows;
        batchIndex++;
      }
      const batch = source.batches[batchIndex];
      const values = source.schema.fields.map((field, columnIndex) =>
        formatCSVValue(batch.getChildAt(columnIndex)!, row - batchOffset, field.type)
      );
      for (const [columnIndex, field] of derivedFields.entries())
        values.push(formatCSVValue(derived!.table.getChildAt(columnIndex)!, index, field.type));
      collector.write(encoder.encode([...values.map(quoteCSV), String(row)].join(',') + '\r\n'));
      if ((index + 1) % 256 === 0) await yieldExport(options.assertCurrent);
    }
  }
  options.assertCurrent();
  const bytes = collector.finishBytes();
  options.assertCurrent();
  return {
    bytes,
    format: options.format,
    rowCount: sourceRows.length,
    sourceRowColumn,
    recordBatchCount,
    ...(derivedColumns ? {derivedColumns} : {})
  };
}

export class BoundedByteCollector extends arrow.AsyncByteQueue {
  private readonly chunks: Uint8Array[] = [];
  private byteLength = 0;
  constructor(private readonly maximumBytes: number) {
    super();
    validateBudget(maximumBytes);
  }
  override write(value: Parameters<arrow.AsyncByteQueue['write']>[0]): void {
    if (!(value instanceof Uint8Array)) throw new Error('Expected an Arrow byte chunk');
    if (this.byteLength + value.byteLength > this.maximumBytes)
      throw new Error('Source export exceeds the download byte budget');
    this.byteLength += value.byteLength;
    this.chunks.push(value);
  }
  finishBytes(): Uint8Array<ArrayBuffer> {
    const bytes = new Uint8Array(this.byteLength);
    let offset = 0;
    for (const chunk of this.chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    this.chunks.length = 0;
    return bytes;
  }
}

function validateBudget(value: number): void {
  if (!Number.isSafeInteger(value) || value < 1 || !Number.isSafeInteger(value * 2))
    throw new Error('Invalid source export byte budget');
}

export function quoteCSV(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

export function formatCSVValue(vector: arrow.Vector, row: number, type: arrow.DataType): string {
  if (!vector.isValid(row)) return '';
  if (arrow.DataType.isDictionary(vector.type)) {
    const original = resolveOriginalColumnRow(vector, row);
    return original ? formatCSVValue(original.vector, original.row, original.vector.type) : '';
  }
  if (arrow.DataType.isTimestamp(type)) {
    return formatTimestampTicks(readOriginalTimestampTicks(vector, row), type.unit);
  }
  const value: unknown = vector.get(row);
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'bigint' || typeof value === 'number' || typeof value === 'boolean')
    return String(value);
  if (value instanceof Date) return value.toISOString();
  if (ArrayBuffer.isView(value))
    return JSON.stringify(
      Array.from(new Uint8Array(value.buffer, value.byteOffset, value.byteLength))
    );
  return JSON.stringify(value, (_key, item) => (typeof item === 'bigint' ? item.toString() : item));
}

async function yieldExport(assertCurrent: () => void): Promise<void> {
  await new Promise<void>(resolve => setTimeout(resolve, 0));
  assertCurrent();
}
