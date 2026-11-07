// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerComputeSession, KeplerSessionExplorer} from './compute-session';

type ExportValues = Float32Array | Int32Array | Uint32Array;
export type KeplerSelectionExportOptions = {
  columns: readonly string[];
  /** Includes count queries and all requested column/validity/identity payloads. */
  maximumReadbackBytes: number;
  /** Retained typed output plus two bounded page payloads; excludes later text serialization. */
  maximumCpuBytes: number;
  signal?: AbortSignal;
} & ({mode: 'page'; offset: number; limit: number} | {mode: 'all-selected'; maximumRows: number});

export type KeplerSelectionExport = {
  mode: 'page' | 'all-selected';
  representation: 'prepared-gpu-columns';
  datasetId: string;
  datasetRevision: number;
  selectionRevision: number;
  offset: number;
  selectedCount: number;
  rowCount: number;
  readbackBytes: number;
  retainedBytes: number;
  schema: readonly {name: string; format: string; nullable: boolean}[];
  dictionaries: Readonly<
    Record<string, {readonly values: readonly string[]; readonly ordered: boolean}>
  >;
  sourceRowIndices: Uint32Array;
  columns: Record<string, ExportValues>;
  validity: Record<string, Uint32Array>;
  /** Drop all result/array aliases when releasing its CPU memory reservation. */
  destroy(): void;
};

/**
 * Explicit bounded GPU materialization. It never uses Kepler CPU filteredIndex or silently exports
 * an inspection page as a full dataset. No result publishes after cancellation or revision changes.
 */
export async function materializeKeplerSelection(
  session: KeplerComputeSession,
  explorer: KeplerSessionExplorer,
  options: KeplerSelectionExportOptions
): Promise<KeplerSelectionExport> {
  const names = [...options.columns];
  if (
    new Set(names).size !== names.length ||
    names.some(name => !explorer.dataset.value.columns[name])
  )
    throw new Error('Export requires unique prepared GPU columns');
  for (const value of [options.maximumReadbackBytes, options.maximumCpuBytes]) validateCount(value);
  if (options.maximumReadbackBytes < 4) throw new Error('Export readback budget is too small');
  if (options.mode === 'page') {
    validateCount(options.offset);
    validateCount(options.limit);
  } else validateCount(options.maximumRows);
  const revision = explorer.currentRevision;
  const dataset = explorer.dataset;
  const assertCurrent = () => {
    options.signal?.throwIfAborted();
    if (
      !explorer.isCurrent ||
      explorer.currentRevision !== revision ||
      session.registry.getDataset(dataset.id) !== dataset
    )
      throw new Error('Export dataset or selection is stale');
  };
  assertCurrent();
  const count = await explorer.readSelectedCount();
  assertCurrent();
  if (!count || count.revision !== revision || count.datasetRevision !== dataset.revision)
    throw new Error('Export count is unavailable; submit the current selection before retrying');
  const selectedCount = count.selectedCount;
  const offset = options.mode === 'page' ? options.offset : 0;
  const rowCount =
    options.mode === 'page'
      ? Math.min(options.limit, Math.max(0, selectedCount - offset))
      : selectedCount;
  if (options.mode === 'all-selected' && rowCount > options.maximumRows)
    throw new Error('Full selected export exceeds its row budget');
  const nullable = names.filter(name => dataset.value.analytics.validity[name]);
  const bytesPerRow = (1 + names.length + nullable.length) * 4;
  const pageSize = Math.min(
    explorer.maximumInspectionPageSize,
    Math.floor((explorer.maximumInspectionReadbackBytes - 4) / bytesPerRow)
  );
  if (rowCount && pageSize < 1)
    throw new Error('Explorer inspection budget cannot fit one export row');
  const readbackBytes =
    4 + rowCount * bytesPerRow + (rowCount ? Math.ceil(rowCount / pageSize) * 4 : 0);
  const retainedBytes = rowCount * bytesPerRow;
  const workingBytes = retainedBytes + Math.min(rowCount, pageSize) * bytesPerRow * 2;
  if (!Number.isSafeInteger(workingBytes) || workingBytes > options.maximumCpuBytes)
    throw new Error('Export exceeds its CPU materialization budget');
  if (readbackBytes > options.maximumReadbackBytes)
    throw new Error('Export exceeds its readback budget');
  const reservation = session.registry.reserveMemory(workingBytes, 'cpu', 'readback');
  let adopted = false;
  try {
    const indices = new Uint32Array(rowCount);
    const columns: Record<string, ExportValues> = Object.create(null);
    const validity: Record<string, Uint32Array> = Object.create(null);
    for (const name of names) {
      const format = dataset.value.columns[name].format;
      columns[name] =
        format === 'float32'
          ? new Float32Array(rowCount)
          : format === 'sint32'
            ? new Int32Array(rowCount)
            : new Uint32Array(rowCount);
    }
    for (const name of nullable) validity[name] = new Uint32Array(rowCount);
    for (let copied = 0; copied < rowCount; copied += pageSize) {
      assertCurrent();
      const length = Math.min(pageSize, rowCount - copied);
      const page = await explorer.readPage(offset + copied, length, names);
      assertCurrent();
      if (
        !page ||
        page.revision !== revision ||
        page.datasetRevision !== dataset.revision ||
        page.sourceRowIndices.length !== length
      )
        throw new Error('Export page is unavailable or stale');
      indices.set(page.sourceRowIndices, copied);
      for (const name of names) columns[name].set(page.columns[name], copied);
      for (const name of nullable) validity[name].set(page.validity[name], copied);
    }
    assertCurrent();
    let destroyed = false;
    const result: KeplerSelectionExport = {
      mode: options.mode,
      representation: 'prepared-gpu-columns',
      datasetId: dataset.id,
      datasetRevision: dataset.revision,
      selectionRevision: revision,
      offset,
      selectedCount,
      rowCount,
      readbackBytes,
      retainedBytes,
      schema: names.map(name => {
        const format = dataset.value.columns[name].format;
        if (!format) throw new Error('Export column format is unavailable');
        return {name, format, nullable: nullable.includes(name)};
      }),
      dictionaries: dataset.value.prepared.dictionaries,
      sourceRowIndices: indices,
      columns,
      validity,
      destroy() {
        if (destroyed) return;
        destroyed = true;
        result.sourceRowIndices = new Uint32Array();
        result.columns = {};
        result.validity = {};
        reservation.release();
      }
    };
    adopted = true;
    return result;
  } finally {
    if (!adopted) reservation.release();
  }
}

function validateCount(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error('Invalid export bound');
}
