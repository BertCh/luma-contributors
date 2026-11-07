// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {KeplerComputeSession, KeplerSessionExplorer} from '@luma-contributors/kepler';
import {exportSelectedSource} from './source-export';
import {
  formatCsvHeader,
  formatTableCsvRows,
  makeNativeExportPage,
  planNativeExport,
  type NativeExportDataset,
  type NativeExportSelection
} from './native-source-export';

export type StreamingExportFormat = 'arrow' | 'csv';

export type StreamingExportProgress = {
  rowsWritten: number;
  totalRows: number;
  bytesWritten: number;
  pagesWritten: number;
};

/** Destination for chunks; a file handle when the browser allows it, otherwise a bounded Blob. */
export type StreamingSink = {
  kind: 'file-system' | 'blob';
  write(chunk: Uint8Array): Promise<void>;
  close(): Promise<void>;
  abort(reason: unknown): Promise<void>;
};

/** Pages are read lazily; every page of one export shares one schema. */
export type StreamingPageSource = {
  /** Called once; returns the row count to write. */
  getTotalRows(): Promise<number> | number;
  readPage(offset: number, limit: number, signal: AbortSignal | undefined): Promise<arrow.Table>;
};

export const STREAMING_EXPORT_DEFAULTS = Object.freeze({
  pageRows: 25_000,
  /** In-memory fallback ceiling: a Blob holds every chunk, so it needs its own explicit limit. */
  maximumBlobBytes: 512 * 1024 * 1024
});

type SaveFilePicker = (options: {
  suggestedName: string;
  types?: {description: string; accept: Record<string, string[]>}[];
}) => Promise<{createWritable(): Promise<WritableStream<Uint8Array> & {close(): Promise<void>}>}>;

/** Prefer the File System Access API; Blob download is the fallback. Must run from a user gesture. */
export async function openStreamingSink(
  filename: string,
  format: StreamingExportFormat,
  options: {
    maximumBlobBytes?: number;
    /** Test hook and escape hatch: skip showSaveFilePicker. */
    preferBlob?: boolean;
    download?: (filename: string, blob: Blob) => void;
  } = {}
): Promise<StreamingSink> {
  const picker = (globalThis as {showSaveFilePicker?: SaveFilePicker}).showSaveFilePicker;
  const mimeType = format === 'arrow' ? 'application/vnd.apache.arrow.stream' : 'text/csv';
  if (picker && !options.preferBlob) {
    const handle = await picker({
      suggestedName: filename,
      types: [
        {
          description: format === 'arrow' ? 'Arrow IPC stream' : 'CSV',
          accept: {[mimeType]: [format === 'arrow' ? '.arrows' : '.csv']}
        }
      ]
    });
    const writable = await handle.createWritable();
    const writer = writable.getWriter();
    return {
      kind: 'file-system',
      write: chunk => writer.write(chunk),
      close: () => writer.close(),
      abort: reason => writer.abort(reason)
    };
  }
  const maximumBytes = options.maximumBlobBytes ?? STREAMING_EXPORT_DEFAULTS.maximumBlobBytes;
  const chunks: Uint8Array[] = [];
  let byteLength = 0;
  return {
    kind: 'blob',
    async write(chunk) {
      byteLength += chunk.byteLength;
      // Without a file picker the whole output must fit in memory.
      if (byteLength > maximumBytes)
        throw new Error(
          `Streaming download exceeds the ${Math.round(maximumBytes / (1024 * 1024))} MiB in-memory fallback; use a browser with the File System Access API`
        );
      chunks.push(Uint8Array.from(chunk));
    },
    async close() {
      const blob = new Blob(chunks as BlobPart[], {type: mimeType});
      chunks.length = 0;
      (options.download ?? downloadBlob)(filename, blob);
    },
    async abort() {
      chunks.length = 0;
    }
  };
}

function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

class ChunkCollector extends arrow.AsyncByteQueue {
  private chunks: Uint8Array[] = [];
  override write(value: Parameters<arrow.AsyncByteQueue['write']>[0]): void {
    if (!(value instanceof Uint8Array)) throw new Error('Expected an Arrow byte chunk');
    this.chunks.push(value);
  }
  drain(): Uint8Array[] {
    const drained = this.chunks;
    this.chunks = [];
    return drained;
  }
}

/**
 * Writes pages to the sink with bounded memory: one page plus one encoded chunk at a time. The
 * export stops at the next page boundary when `signal` aborts, and the sink is aborted so no
 * partial download is published.
 */
export async function runStreamingExport(props: {
  source: StreamingPageSource;
  format: StreamingExportFormat;
  sink: StreamingSink;
  pageRows?: number;
  signal?: AbortSignal;
  onProgress?(progress: StreamingExportProgress): void;
}): Promise<StreamingExportProgress> {
  const pageRows = props.pageRows ?? STREAMING_EXPORT_DEFAULTS.pageRows;
  const progress: StreamingExportProgress = {
    rowsWritten: 0,
    totalRows: 0,
    bytesWritten: 0,
    pagesWritten: 0
  };
  const send = async (chunk: Uint8Array) => {
    await props.sink.write(chunk);
    progress.bytesWritten += chunk.byteLength;
  };
  try {
    props.signal?.throwIfAborted();
    progress.totalRows = await props.source.getTotalRows();
    const encoder = new TextEncoder();
    const collector = new ChunkCollector();
    let writer: arrow.RecordBatchStreamWriter | undefined;
    let schema: arrow.Schema | undefined;
    for (let offset = 0; offset === 0 || offset < progress.totalRows; offset += pageRows) {
      props.signal?.throwIfAborted();
      const page = await props.source.readPage(offset, pageRows, props.signal);
      props.signal?.throwIfAborted();
      if (!schema) {
        schema = page.schema;
        if (props.format === 'arrow') {
          writer = new arrow.RecordBatchStreamWriter();
          writer.reset(collector, schema);
        } else await send(encoder.encode(formatCsvHeader(schema)));
      } else if (
        page.schema.fields.length !== schema.fields.length ||
        page.schema.fields.some(
          (field, index) => field.type.toString() !== schema!.fields[index].type.toString()
        )
      )
        // Changing column types between pages cannot be written into one stream.
        throw new Error('Export page schema changed between pages');
      if (props.format === 'arrow') {
        for (const batch of page.batches) writer!.write(new arrow.RecordBatch(schema, batch.data));
        for (const chunk of collector.drain()) await send(chunk);
      } else await send(encoder.encode(formatTableCsvRows(page)));
      progress.rowsWritten += page.numRows;
      progress.pagesWritten++;
      props.onProgress?.({...progress});
      // Let the browser paint progress and honor Cancel between bounded pages.
      await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
    if (writer) {
      writer.close();
      for (const chunk of collector.drain()) await send(chunk);
    }
    props.signal?.throwIfAborted();
    await props.sink.close();
    return progress;
  } catch (error) {
    await props.sink.abort(error).catch(() => undefined);
    throw error;
  }
}

/** Native tables are paged from the Kepler dataContainer (never GPU readback). */
export function makeNativePageSource(
  dataset: NativeExportDataset,
  selection: NativeExportSelection
): StreamingPageSource {
  const revision = dataset.dataRevision;
  let plan: ReturnType<typeof planNativeExport> | undefined;
  return {
    getTotalRows() {
      plan = planNativeExport(dataset, selection);
      return plan.rowIndices.length;
    },
    async readPage(offset, limit) {
      if (!plan || dataset.dataRevision !== revision)
        throw new Error('Native dataset changed during streaming export');
      return makeNativeExportPage(plan, offset, limit);
    }
  };
}

/**
 * Resident datasets read bounded GPU pages. Each page is one explicit selection export; there is
 * no whole-selection buffer, so the output may exceed the non-streaming byte and row limits.
 */
export function makeResidentPageSource(
  session: KeplerComputeSession,
  explorer: KeplerSessionExplorer,
  limits: {maximumPageReadbackBytes: number; maximumPageOutputBytes: number} = {
    maximumPageReadbackBytes: 16 * 1024 * 1024,
    maximumPageOutputBytes: 64 * 1024 * 1024
  }
): StreamingPageSource {
  let totalRows = 0;
  const revision = explorer.currentRevision;
  const readOne = async (offset: number, limit: number, signal: AbortSignal | undefined) => {
    if (explorer.currentRevision !== revision || !explorer.isCurrent)
      throw new Error('Selection changed during streaming export');
    const result = await exportSelectedSource(session, explorer, {
      mode: 'page',
      offset,
      limit,
      format: 'arrow',
      maximumReadbackBytes: limits.maximumPageReadbackBytes,
      maximumCpuBytes: limits.maximumPageOutputBytes,
      maximumOutputBytes: limits.maximumPageOutputBytes,
      ...(signal ? {signal} : {})
    });
    try {
      totalRows = result.selectedCount;
      return arrow.tableFromIPC(result.bytes);
    } finally {
      result.destroy();
    }
  };
  return {
    async getTotalRows() {
      // The count is part of every page response; the first page also supplies the schema.
      const first = await readOne(0, 1, undefined);
      void first;
      return totalRows;
    },
    readPage: readOne
  };
}
