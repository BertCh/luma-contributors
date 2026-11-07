// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {GPUTable} from '@luma-contributors/graph/gpu-tables';
import {getGPUPartitionMemoryPlan} from '@luma-contributors/graph/gpu-spatial-analysis';
import {
  getGPUPartitionDescriptorFromArrowTable,
  type PreparedArrowAnalyticsTable
} from '@luma-contributors/graph/arrow';
import {
  Dictionary,
  Field,
  Float32,
  Int32,
  RecordBatch,
  Schema,
  Struct,
  Table,
  Uint32,
  Utf8,
  makeData,
  vectorFromArray,
  type Data,
  type Vector,
  type DataType
} from 'apache-arrow';
import type {GPUResidentColumns} from '../exploration/gpu-resident-explorer';
import type {KeplerDatasetRequest, KeplerMemoryReservation} from '../runtime/dataset-registry';
import type {KeplerArrowDataset} from './compute-session';

/** Scalar GPU storage formats a resident dataset can carry. */
export type KeplerResidentFormat = 'float32' | 'sint32' | 'uint32';

/** Default rows per lazy readback page. A page never crosses a source chunk boundary. */
export const DEFAULT_KEPLER_RESIDENT_PAGE_ROWS = 1024;

/** Schema of one GPU-resident column; `dictionary` marks uint32 codes with UTF-8 labels. */
export type KeplerResidentColumnSchema = {
  name: string;
  format: KeplerResidentFormat;
  /** Labels for uint32 category codes. Out-of-range codes read as null. */
  dictionary?: readonly string[];
  metadata?: Readonly<Record<string, string>>;
};

export type KeplerResidentDatasetOptions = {
  id: string;
  schema: readonly KeplerResidentColumnSchema[];
  /** One GPU vector per schema column. Chunk boundaries become source batch boundaries. */
  columns: GPUResidentColumns;
  /** Present for columns that can contain nulls; 1 marks a valid row. */
  validity: Readonly<Record<string, GPUVector<'uint32'> | undefined>>;
  schemaMetadata?: Readonly<Record<string, string>>;
  /**
   * Exact time words for schema columns that hold time values (for example sint32 epoch seconds
   * tagged with `luma.gl.time.format`). One chunk per resident chunk; the dataset uploads them and
   * serves native time filters from them. The schema column stays for display and SQL.
   */
  timeColumns?: Readonly<Record<string, PreparedArrowAnalyticsTable['timeColumns'][string]>>;
  /** Rows per lazy readback page. Defaults to DEFAULT_KEPLER_RESIDENT_PAGE_ROWS. */
  pageRows?: number;
  request: KeplerDatasetRequest;
  /** Must cover estimateKeplerResidentRowCacheByteLength(); consumed during registration. */
  cpuReservation: KeplerMemoryReservation;
  gpuReservation: KeplerMemoryReservation;
  /** Owns dedicated output buffers, not borrowed source chunks. Called only on dataset disposal. */
  destroy(): void;
};

/**
 * Upper bound of retained lazy-cache memory. Typed arrays are allocated zero-filled, so physical
 * pages are committed only when a row range is first read back; the reservation is the ceiling.
 */
export function estimateKeplerResidentRowCacheByteLength(
  schema: readonly KeplerResidentColumnSchema[],
  chunkLengths: readonly number[]
): number {
  let byteLength = 0;
  for (const length of chunkLengths)
    byteLength += schema.length * (length * 4 + Math.ceil(length / 8));
  return (
    byteLength +
    schema.reduce(
      (total, column) =>
        total + (column.dictionary?.reduce((bytes, label) => bytes + label.length * 2, 0) ?? 0),
      0
    )
  );
}

type LazyColumnState = {
  schema: KeplerResidentColumnSchema;
  /** Per chunk typed storage and Arrow validity bitmap, filled page by page. */
  values: (Float32Array | Int32Array | Uint32Array)[];
  bitmaps: Uint8Array[];
  data: Data[];
};

/**
 * Lazy, bounded, accounted CPU view of GPU-resident columns.
 *
 * `table` has the dataset's real schema, row count and batch layout, but cell storage starts as
 * all-null. `ensureRows()` reads only the pages containing the requested rows from the GPU and
 * fills them in place, so Arrow consumers see real values for rows that were requested and null
 * for rows that were never needed. No full-column readback ever happens.
 */
export class KeplerResidentRows {
  readonly table: Table;
  readonly rowCount: number;
  readonly pageRows: number;
  private readonly columns: LazyColumnState[];
  private readonly chunkLengths: readonly number[];
  private readonly chunkOffsets: readonly number[];
  private readonly loadedPages: Uint8Array[];
  private readonly pending = new Map<string, Promise<void>>();
  private readonly listeners = new Set<() => void>();
  private readonly gpuColumns: GPUResidentColumns;
  private readonly gpuValidity: Readonly<Record<string, GPUVector<'uint32'> | undefined>>;
  private readonly device: Device;
  private queue: Promise<void> = Promise.resolve();
  private closed = false;
  private domainValues: Record<string, readonly [number, number] | null> = {};
  private readbackByteLengthValue = 0;
  private loadedRowCountValue = 0;

  constructor(
    device: Device,
    options: Pick<
      KeplerResidentDatasetOptions,
      'schema' | 'columns' | 'validity' | 'schemaMetadata' | 'pageRows'
    >
  ) {
    this.device = device;
    this.gpuColumns = options.columns;
    this.gpuValidity = options.validity;
    this.pageRows = options.pageRows ?? DEFAULT_KEPLER_RESIDENT_PAGE_ROWS;
    if (!Number.isInteger(this.pageRows) || this.pageRows < 1)
      throw new Error('Resident page rows must be a positive integer');
    const first = options.columns[options.schema[0]?.name];
    this.chunkLengths = first ? first.data.map(chunk => chunk.length) : [];
    let offset = 0;
    this.chunkOffsets = this.chunkLengths.map(length => {
      const chunkOffset = offset;
      offset += length;
      return chunkOffset;
    });
    this.rowCount = offset;
    this.loadedPages = this.chunkLengths.map(
      length => new Uint8Array(Math.ceil(length / this.pageRows))
    );
    this.columns = options.schema.map(schema => ({
      schema,
      values: [],
      bitmaps: [],
      data: []
    }));
    const fields: Field[] = [];
    const dictionaryTypes: DataType[] = [];
    for (const column of this.columns) {
      const {schema} = column;
      const type: DataType = schema.dictionary
        ? new Dictionary(new Utf8(), new Uint32())
        : schema.format === 'float32'
          ? new Float32()
          : schema.format === 'sint32'
            ? new Int32()
            : new Uint32();
      dictionaryTypes.push(type);
      fields.push(
        new Field(schema.name, type, true, new Map(Object.entries(schema.metadata ?? {})))
      );
    }
    const arrowSchema = new Schema(fields, new Map(Object.entries(options.schemaMetadata ?? {})));
    const labels = new Map<string, Vector<Utf8>>();
    const batches: RecordBatch[] = [];
    for (const length of this.chunkLengths) {
      const children: Data[] = [];
      for (const [columnIndex, column] of this.columns.entries()) {
        const type = dictionaryTypes[columnIndex];
        const values =
          column.schema.format === 'float32'
            ? new Float32Array(length)
            : column.schema.format === 'sint32'
              ? new Int32Array(length)
              : new Uint32Array(length);
        const bitmap = new Uint8Array(Math.ceil(length / 8));
        let dictionary: Vector<Utf8> | undefined;
        if (column.schema.dictionary) {
          dictionary = labels.get(column.schema.name);
          if (!dictionary) {
            dictionary = vectorFromArray([...column.schema.dictionary], new Utf8());
            labels.set(column.schema.name, dictionary);
          }
        }
        // nullCount -1 recomputes lazily from the bitmap, which pages fill in as they arrive.
        const data = makeData({
          type,
          length,
          nullCount: -1,
          nullBitmap: bitmap,
          data: values,
          ...(dictionary ? {dictionary} : {})
        } as Parameters<typeof makeData>[0]);
        column.values.push(values);
        column.bitmaps.push(bitmap);
        column.data.push(data);
        children.push(data);
      }
      batches.push(
        new RecordBatch(
          arrowSchema,
          makeData({type: new Struct(fields), length, nullCount: 0, children})
        )
      );
    }
    this.table = batches.length ? new Table(arrowSchema, batches) : new Table(arrowSchema);
  }

  /** Bytes read from the GPU so far; grows by whole pages only. */
  get readbackByteLength(): number {
    return this.readbackByteLengthValue;
  }

  get loadedRowCount(): number {
    return this.loadedRowCountValue;
  }

  /** Per-column [minimum, maximum] over valid rows, set by the producer; null when empty. */
  get domains(): Readonly<Record<string, readonly [number, number] | null>> {
    return this.domainValues;
  }

  setDomains(domains: Readonly<Record<string, readonly [number, number] | null>>): void {
    this.domainValues = {...domains};
  }

  isRowLoaded(row: number): boolean {
    const location = this.locate(row);
    return Boolean(
      location && this.loadedPages[location.chunk][Math.floor(location.row / this.pageRows)]
    );
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Read the pages that contain these rows. Rows already loaded cost nothing. */
  async ensureRows(rows: readonly number[]): Promise<void> {
    const wanted = new Map<string, {chunk: number; page: number}>();
    for (const row of rows) {
      const location = this.locate(row);
      if (!location) throw new Error('Resident row is outside the dataset');
      const page = Math.floor(location.row / this.pageRows);
      if (!this.loadedPages[location.chunk][page])
        wanted.set(`${location.chunk}:${page}`, {chunk: location.chunk, page});
    }
    await Promise.all(
      [...wanted].map(([key, {chunk, page}]) => {
        let request = this.pending.get(key);
        if (!request) {
          const load = this.queue.then(() => this.loadPage(chunk, page));
          // A failed read must not block later pages.
          this.queue = load.catch(() => undefined);
          request = load.finally(() => this.pending.delete(key));
          this.pending.set(key, request);
        }
        return request;
      })
    );
  }

  /** Pending and future page reads are dropped; the GPU buffers belong to the dataset. */
  close(): void {
    this.closed = true;
    this.listeners.clear();
  }

  private locate(row: number): {chunk: number; row: number} | null {
    if (!Number.isInteger(row) || row < 0 || row >= this.rowCount) return null;
    let low = 0;
    let high = this.chunkOffsets.length - 1;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      if (this.chunkOffsets[middle] <= row) low = middle;
      else high = middle - 1;
    }
    return {chunk: low, row: row - this.chunkOffsets[low]};
  }

  private async loadPage(chunk: number, page: number): Promise<void> {
    if (this.closed || this.device.isLost || this.loadedPages[chunk][page]) return;
    const start = page * this.pageRows;
    const count = Math.min(this.pageRows, this.chunkLengths[chunk] - start);
    const reads: Promise<Uint8Array>[] = [];
    const sources: (GPUData | undefined)[] = [];
    for (const column of this.columns) {
      for (const vector of [
        this.gpuColumns[column.schema.name],
        this.gpuValidity[column.schema.name]
      ]) {
        const data = vector?.data[chunk];
        sources.push(data);
        reads.push(
          data
            ? Promise.resolve().then(() =>
                (data.buffer as Buffer).readAsync(data.byteOffset + start * 4, count * 4)
              )
            : Promise.resolve(new Uint8Array(0))
        );
      }
    }
    const results = await Promise.all(reads);
    if (this.closed || this.device.isLost) return;
    for (const [columnIndex, column] of this.columns.entries()) {
      const bytes = results[columnIndex * 2];
      const validityBytes = results[columnIndex * 2 + 1];
      const words = new Uint32Array(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + count * 4)
      );
      const valid = sources[columnIndex * 2 + 1]
        ? new Uint32Array(
            validityBytes.buffer.slice(
              validityBytes.byteOffset,
              validityBytes.byteOffset + count * 4
            )
          )
        : undefined;
      const target = column.values[chunk];
      const bitmap = column.bitmaps[chunk];
      const labelCount = column.schema.dictionary?.length;
      for (let index = 0; index < count; index++) {
        const row = start + index;
        const isValid =
          (valid ? valid[index] !== 0 : true) &&
          (labelCount === undefined || words[index] < labelCount);
        // Raw words are reinterpreted per format; the copy above is four-byte aligned.
        if (column.schema.format === 'float32')
          target[row] = new Float32Array(words.buffer, index * 4, 1)[0];
        else if (column.schema.format === 'sint32')
          target[row] = new Int32Array(words.buffer, index * 4, 1)[0];
        else target[row] = words[index];
        if (isValid) bitmap[row >> 3] |= 1 << (row & 7);
        else bitmap[row >> 3] &= ~(1 << (row & 7));
      }
      // Force Arrow to recount nulls from the updated bitmap on next access.
      (column.data[chunk] as unknown as {_nullCount: number})._nullCount = -1;
    }
    this.readbackByteLengthValue += results.reduce((total, bytes) => total + bytes.byteLength, 0);
    this.loadedPages[chunk][page] = 1;
    this.loadedRowCountValue += count;
    for (const listener of [...this.listeners]) listener();
  }
}

/** Prepared-table view of a resident dataset: real batch layout and dictionaries, lazy cells. */
export function makeResidentPreparedTable(
  rows: KeplerResidentRows,
  schema: readonly KeplerResidentColumnSchema[],
  timeColumns: PreparedArrowAnalyticsTable['timeColumns'] = {}
): PreparedArrowAnalyticsTable {
  let offset = 0;
  return {
    table: rows.table,
    sourceRowCount: rows.rowCount,
    dictionaries: Object.fromEntries(
      schema.flatMap(column =>
        column.dictionary ? [[column.name, {values: [...column.dictionary], ordered: false}]] : []
      )
    ),
    timeColumns,
    conversions: [],
    sourceBatches: rows.table.batches.map((batch, index) => {
      const identity = {
        sourceBatchIndex: index,
        sourceRowIndexOffset: offset,
        sourceRowCount: batch.numRows
      };
      offset += batch.numRows;
      return identity;
    })
  };
}

/** Borrow caller-owned vectors as dataset analytics; the dataset destroys them via options.destroy. */
export function makeResidentAnalytics(
  device: Device,
  options: KeplerResidentDatasetOptions,
  rows: KeplerResidentRows,
  prepared: PreparedArrowAnalyticsTable
): KeplerArrowDataset['analytics'] {
  if (device.isLost) throw new Error('Cannot adopt outputs from a lost device');
  const names = options.schema.map(column => column.name);
  if (
    !options.id ||
    !names.length ||
    new Set(names).size !== names.length ||
    Object.keys(options.columns).length !== names.length ||
    Object.keys(options.validity).some(name => !names.includes(name))
  )
    throw new Error('Resident schema and GPU columns must match');
  const buffers = new Set<Buffer>();
  const validateVector = (vector: GPUVector | undefined, format: string): void => {
    if (
      !vector ||
      vector.format !== format ||
      vector.length !== rows.rowCount ||
      vector.data.length !== rows.table.batches.length
    )
      throw new Error('Resident GPU columns must match the schema format and batches');
    for (const [index, chunk] of vector.data.entries()) {
      if (
        chunk.length !== rows.table.batches[index].numRows ||
        !(chunk.buffer instanceof Buffer) ||
        chunk.buffer.device !== device ||
        chunk.buffer.destroyed ||
        chunk.byteStride !== 4 ||
        !(chunk.buffer.usage & Buffer.STORAGE) ||
        !(chunk.buffer.usage & Buffer.COPY_SRC)
      )
        throw new Error('Resident GPU chunks require readable scalar storage on this device');
      if (buffers.has(chunk.buffer))
        throw new Error('Resident output columns require disjoint buffers');
      buffers.add(chunk.buffer);
    }
  };
  for (const column of options.schema) {
    validateVector(options.columns[column.name], column.dictionary ? 'uint32' : column.format);
    if (column.dictionary && column.format !== 'uint32')
      throw new Error('Resident dictionary columns store uint32 codes');
    if (options.validity[column.name]) validateVector(options.validity[column.name], 'uint32');
  }
  const borrow = <Format extends KeplerResidentFormat>(
    name: string,
    vector: GPUVector<Format>
  ): GPUVector<Format> =>
    new GPUVector({
      name,
      type: 'data',
      format: vector.format,
      ownsData: true,
      data: vector.data.map(
        chunk =>
          new GPUData({
            format: vector.format,
            buffer: chunk.buffer,
            length: chunk.length,
            byteOffset: chunk.byteOffset,
            ownsBuffer: false
          })
      )
    });
  const borrowed: GPUResidentColumns = Object.fromEntries(
    Object.entries(options.columns).map(([name, vector]) => [name, borrow(name, vector)])
  );
  const borrowedValidity = Object.fromEntries(
    Object.entries(options.validity).map(([name, vector]) => [
      name,
      vector ? borrow(name, vector) : undefined
    ])
  );
  const table = new GPUTable<Record<string, KeplerResidentFormat>>({vectors: borrowed});
  for (const schema of [table.schema, ...table.batches.map(batch => batch.schema)]) {
    for (const field of schema.fields) {
      const arrowField = rows.table.schema.fields.find(candidate => candidate.name === field.name)!;
      field.nullable = Boolean(options.validity[field.name]);
      field.metadata = new Map(arrowField.metadata);
    }
  }
  const partitioning = getGPUPartitionDescriptorFromArrowTable(rows.table);
  return {
    table,
    validity: borrowedValidity,
    dictionaries: {...prepared.dictionaries},
    nullCounts: Object.fromEntries(names.map(name => [name, rows.table.batches.map(() => 0)])),
    partitioning,
    partitionMemory: getGPUPartitionMemoryPlan(
      partitioning,
      names.reduce((bytes, name) => bytes + 4 + (options.validity[name] ? 4 : 0), 0)
    )
  };
}
