// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  GPUKeplerTableInspector,
  type KeplerComputeSession,
  type KeplerSessionExplorer,
  type KeplerSelectionExportOptions,
  type KeplerTableDerivedSnapshot
} from '@luma-contributors/kepler';
import type {AnalysisRuntime} from './analysis-runtime';
import {BoundedByteCollector, serializeSelectedSource} from './source-export';

import {materializeResidentRows} from './resident-materialization';

import {clamp} from '@math.gl/core';

export type AnalysisExportFormat = 'json' | 'arrow' | 'csv';
type ExportOptions = Omit<KeplerSelectionExportOptions, 'columns' | 'mode'> &
  ({mode: 'page'; offset: number; limit: number} | {mode: 'all-selected'; maximumRows: number}) & {
    format: AnalysisExportFormat;
    maximumOutputBytes: number;
    includeOriginalSource?: boolean;
  };
type Values = Float32Array | Int32Array | Uint32Array;
export type AnalysisExportData = {
  datasetId: string;
  datasetRevision: number;
  selectionRevision: number;
  analysisRevision: string;
  recipe: unknown;
  mode: 'page' | 'all-selected';
  offset: number;
  selectedCount: number;
  sourceRowIndices: Uint32Array;
  columns: Record<string, Values>;
  validity: Record<string, Uint32Array>;
};

/** Explicit selected-result readback. Its dedicated inspector cannot supersede native table reads. */
export function exportSelectedAnalysis(
  session: KeplerComputeSession,
  explorer: KeplerSessionExplorer,
  analysis: AnalysisRuntime,
  options: ExportOptions
) {
  return analysis.withResultSnapshot(snapshot => {
    const recipe: unknown = JSON.parse(analysis.serializeRecipe());
    return readAnalysis(session, explorer, snapshot, recipe, options);
  });
}

async function readAnalysis(
  session: KeplerComputeSession,
  explorer: KeplerSessionExplorer,
  snapshot: KeplerTableDerivedSnapshot,
  recipe: unknown,
  options: ExportOptions
) {
  const dataset = explorer.dataset;
  const selectionRevision = explorer.currentRevision;
  const assertCurrent = () => {
    options.signal?.throwIfAborted();
    if (
      !snapshot.isCurrent() ||
      !explorer.isCurrent ||
      explorer.currentRevision !== selectionRevision ||
      session.registry.getDataset(dataset.id) !== dataset
    )
      throw new Error('Analysis export dataset, selection or result is stale');
  };
  for (const bound of [
    options.maximumReadbackBytes,
    options.maximumCpuBytes,
    options.maximumOutputBytes,
    ...(options.mode === 'page' ? [options.offset, options.limit] : [options.maximumRows])
  ])
    if (!Number.isSafeInteger(bound) || bound < 0) throw new Error('Invalid analysis export bound');
  if (options.maximumReadbackBytes < 4)
    throw new Error('Analysis export readback budget is too small');
  assertCurrent();
  const count = await explorer.readSelectedCount();
  assertCurrent();
  if (!count || count.revision !== selectionRevision || count.datasetRevision !== dataset.revision)
    throw new Error('Analysis selection count is unavailable');
  const offset = options.mode === 'page' ? options.offset : 0;
  const rowCount =
    options.mode === 'page'
      ? clamp(count.selectedCount - offset, 0, options.limit)
      : count.selectedCount;
  if (options.mode === 'all-selected' && rowCount > options.maximumRows)
    throw new Error('Analysis export exceeds its row budget');
  const names = Object.keys(snapshot.columns);
  const nullable = names.filter(name => snapshot.validity?.[name]);
  const bytesPerRow = (1 + names.length + nullable.length) * 4;
  // The inspector budgets an extra identity per row and two count queries per page.
  const pageSize = Math.min(
    explorer.maximumInspectionPageSize,
    Math.floor((explorer.maximumInspectionReadbackBytes - 8) / (bytesPerRow + 4))
  );
  if (rowCount && pageSize < 1) throw new Error('Analysis inspection budget cannot fit a row');
  const readbackBytes =
    4 + rowCount * bytesPerRow + (rowCount ? Math.ceil(rowCount / pageSize) * 8 : 0);
  const workingBytes = rowCount * bytesPerRow + clamp(pageSize, 0, rowCount) * bytesPerRow * 2;
  if (readbackBytes > options.maximumReadbackBytes)
    throw new Error('Analysis export exceeds its readback budget');
  if (!Number.isSafeInteger(workingBytes) || workingBytes > options.maximumCpuBytes)
    throw new Error('Analysis export exceeds its CPU materialization budget');
  const serializationBytes =
    options.maximumOutputBytes * 2 + Math.ceil(rowCount / 8) * nullable.length;
  if (!Number.isSafeInteger(serializationBytes))
    throw new Error('Invalid analysis serialization budget');
  const reservation = session.registry.reserveMemory(
    workingBytes + serializationBytes,
    'cpu',
    'output'
  );
  let inspector: GPUKeplerTableInspector | undefined;
  let adopted = false;
  try {
    inspector = new GPUKeplerTableInspector({
      session,
      explorer,
      sortMemoryBudgetBytes: 0,
      getDerivedColumns: () => snapshot
    });
    const data: AnalysisExportData = {
      datasetId: dataset.id,
      datasetRevision: dataset.revision,
      selectionRevision,
      analysisRevision: snapshot.revision,
      recipe,
      mode: options.mode,
      offset,
      selectedCount: count.selectedCount,
      sourceRowIndices: new Uint32Array(rowCount),
      columns: Object.fromEntries(
        names.map(name => [
          name,
          snapshot.columns[name].format === 'float32'
            ? new Float32Array(rowCount)
            : snapshot.columns[name].format === 'sint32'
              ? new Int32Array(rowCount)
              : new Uint32Array(rowCount)
        ])
      ),
      validity: Object.fromEntries(nullable.map(name => [name, new Uint32Array(rowCount)]))
    };
    for (let copied = 0; copied < rowCount; copied += pageSize) {
      assertCurrent();
      const length = Math.min(pageSize, rowCount - copied);
      const page = await inspector.readPage({
        population: 'selected',
        offset: offset + copied,
        limit: length,
        columns: names
      });
      assertCurrent();
      if (
        !page ||
        page.revision !== selectionRevision ||
        page.datasetRevision !== dataset.revision ||
        page.totalCount !== count.selectedCount ||
        page.sourceRowIndices.length !== length
      )
        throw new Error('Analysis export page is unavailable or stale');
      data.sourceRowIndices.set(page.sourceRowIndices, copied);
      for (const name of names) data.columns[name].set(page.columns[name], copied);
      for (const name of nullable) data.validity[name].set(page.validity[name], copied);
    }
    if (options.includeOriginalSource)
      await materializeResidentRows(dataset.value, data.sourceRowIndices, {assertCurrent});
    const serialized = await serializeAnalysisExport(data, {
      ...options,
      assertCurrent,
      ...(options.includeOriginalSource ? {source: dataset.value.source} : {})
    });
    assertCurrent();
    let destroyed = false;
    const result = {
      ...serialized,
      datasetId: dataset.id,
      datasetRevision: dataset.revision,
      selectionRevision,
      rowCount,
      readbackBytes,
      assertCurrent,
      destroy() {
        if (destroyed) return;
        destroyed = true;
        result.bytes = new Uint8Array();
        reservation.release();
      }
    };
    adopted = true;
    return result;
  } finally {
    try {
      await inspector?.destroy();
    } catch (error) {
      reservation.release();
      throw error;
    } finally {
      if (!adopted) reservation.release();
    }
  }
}

/** Serialization preserves scalar types and validity in Arrow; JSON represents nonfinite values as strings. */
export async function serializeAnalysisExport(
  data: AnalysisExportData,
  options: {
    format: AnalysisExportFormat;
    maximumOutputBytes: number;
    assertCurrent(): void;
    source?: arrow.Table;
  }
) {
  options.assertCurrent();
  const names = Object.keys(data.columns);
  const rowCount = data.sourceRowIndices.length;
  if (
    names.some(
      name =>
        data.columns[name].length !== rowCount ||
        (data.validity[name] && data.validity[name].length !== rowCount)
    )
  )
    throw new Error('Analysis export columns must align with source identities');
  let sourceRowColumn = '__luma_source_row_index';
  while (names.includes(sourceRowColumn)) sourceRowColumn += '_';
  const {sourceRowIndices, columns, validity, ...context} = data;
  const metadata = {
    ...context,
    representation: 'derived-gpu-analysis',
    sourceRowColumn,
    schema: names.map(name => ({
      name,
      format:
        columns[name] instanceof Float32Array
          ? 'float32'
          : columns[name] instanceof Int32Array
            ? 'sint32'
            : 'uint32',
      nullable: Boolean(validity[name])
    }))
  };
  if (options.source) {
    if (options.format === 'json')
      throw new Error('Combined original-source export requires Arrow or CSV');
    const derived = new arrow.Table(makeAnalysisVectors(columns, validity, rowCount));
    const serialized = await serializeSelectedSource(
      options.source,
      sourceRowIndices,
      {
        datasetId: data.datasetId,
        datasetRevision: data.datasetRevision,
        selectionRevision: data.selectionRevision,
        selectedCount: data.selectedCount,
        offset: data.offset
      },
      {
        ...options,
        format: options.format,
        derived: {table: derived, metadata}
      }
    );
    return {
      ...serialized,
      metadata: JSON.stringify(
        {
          ...context,
          representation: 'original-source-and-derived-analysis',
          analysis: {...metadata, sourceRowColumn: serialized.sourceRowColumn},
          sourceRowColumn: serialized.sourceRowColumn,
          derivedColumns: serialized.derivedColumns,
          sourceSchema: options.source.schema.fields.map(field => ({
            name: field.name,
            type: field.type.toString(),
            nullable: field.nullable
          }))
        },
        null,
        2
      )
    };
  }
  const collector = new BoundedByteCollector(options.maximumOutputBytes);
  const encoder = new TextEncoder();
  const write = (text: string) => collector.write(encoder.encode(text));
  const valueAt = (name: string, row: number): number | string | null => {
    if (validity[name] && !validity[name][row]) return null;
    const value = columns[name][row];
    return Number.isFinite(value) ? value : String(value);
  };
  if (options.format === 'arrow') {
    const vectors: Record<string, arrow.Vector> = Object.create(null);
    vectors[sourceRowColumn] = arrow.makeVector(sourceRowIndices);
    Object.assign(vectors, makeAnalysisVectors(columns, validity, rowCount));
    const table = new arrow.Table(vectors);
    table.schema.metadata.set('luma:analysis-export', JSON.stringify(metadata));
    const writer = new arrow.RecordBatchStreamWriter();
    writer.reset(collector, table.schema);
    for (const batch of table.batches) writer.write(batch);
    writer.close();
  } else {
    if (options.format === 'json')
      write(`${JSON.stringify({...metadata, nonfiniteEncoding: 'string'}).slice(0, -1)},"rows":[`);
    else write([sourceRowColumn, ...names].map(quoteCSV).join(',') + '\r\n');
    for (let row = 0; row < rowCount; row++) {
      if (options.format === 'json')
        write(
          `${row ? ',' : ''}${JSON.stringify({
            [sourceRowColumn]: sourceRowIndices[row],
            ...Object.fromEntries(names.map(name => [name, valueAt(name, row)]))
          })}`
        );
      else
        write(
          [
            String(sourceRowIndices[row]),
            ...names.map(name => {
              const value = valueAt(name, row);
              return value === null ? '' : String(value);
            })
          ]
            .map(quoteCSV)
            .join(',') + '\r\n'
        );
      if ((row + 1) % 256 === 0) {
        await new Promise<void>(resolve => setTimeout(resolve, 0));
        options.assertCurrent();
      }
    }
    if (options.format === 'json') write(']}');
  }
  options.assertCurrent();
  return {
    bytes: collector.finishBytes(),
    format: options.format,
    sourceRowColumn,
    // CSV carries calculation settings in a companion metadata document.
    metadata: JSON.stringify(metadata, null, 2)
  };
}

function quoteCSV(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

function makeAnalysisVectors(
  columns: Record<string, Values>,
  validity: Record<string, Uint32Array>,
  rowCount: number
): Record<string, arrow.Vector> {
  const vectors: Record<string, arrow.Vector> = Object.create(null);
  for (const name of Object.keys(columns)) {
    const values = columns[name];
    let nullCount = 0;
    const nullBitmap = validity[name] ? new Uint8Array(Math.ceil(rowCount / 8)) : undefined;
    if (nullBitmap)
      for (let row = 0; row < rowCount; row++) {
        if (validity[name][row]) nullBitmap[row >> 3] |= 1 << (row & 7);
        else nullCount++;
      }
    const attributes = {length: rowCount, nullBitmap, nullCount};
    vectors[name] =
      values instanceof Float32Array
        ? arrow.makeVector(arrow.makeData({type: new arrow.Float32(), data: values, ...attributes}))
        : values instanceof Int32Array
          ? arrow.makeVector(arrow.makeData({type: new arrow.Int32(), data: values, ...attributes}))
          : arrow.makeVector(
              arrow.makeData({type: new arrow.Uint32(), data: values, ...attributes})
            );
  }
  return vectors;
}
