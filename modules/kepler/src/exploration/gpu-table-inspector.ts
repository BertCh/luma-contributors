// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {Buffer} from '@luma.gl/core';
import {readAllBuffers} from './read-all-buffers';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, GPUGather, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUDataFrame,
  column,
  literal,
  type CompiledGPUDataFrameGlobalSort,
  type GPUDataFrameQueryParameters
} from '@luma.gl/graph/gpu-dataframe';
import {GPUTable} from '@luma.gl/graph/gpu-tables';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerSessionExplorer
} from '../session/compute-session';
import type {KeplerDatasetLease, KeplerMemoryReservation} from '../runtime/dataset-registry';
import type {GPUResidentPage, GPUResidentScalarFormat} from './gpu-resident-explorer';
import {
  getGPUColumnStatisticsByteLength,
  planGPUColumnStatistics,
  readGPUColumnStatistics,
  DEFAULT_GPU_COLUMN_STATISTICS_BINS,
  type GPUColumnStatistics,
  type GPUColumnStatisticsRoute
} from './gpu-column-statistics';

export type KeplerTablePageRequest = {
  offset: number;
  limit: number;
  columns: readonly string[];
  population: 'all' | 'selected';
  sort?: {column: string; direction: 'ascending' | 'descending'};
};
export type KeplerTablePage = GPUResidentPage & {totalCount: number};
/** Borrowed results must outlive the inspector and preserve source batch/row alignment. */
export type KeplerTableDerivedSnapshot = {
  revision: string;
  columns: Readonly<Record<string, GPUVector<GPUResidentScalarFormat>>>;
  validity?: Readonly<Record<string, GPUVector<'uint32'> | undefined>>;
  isCurrent(): boolean;
};

type SortColumns = {value: GPUResidentScalarFormat; selected: 'uint32'};
type SortProgram = {
  key: string;
  query: CompiledGPUDataFrameGlobalSort<SortColumns>;
  reservation: KeplerMemoryReservation;
  revision: number;
};

/**
 * Explicit asynchronous table inspection. Sorting retains a GPU permutation; only one requested
 * page of row IDs and values returns to the CPU. Source batches and columns remain borrowed.
 * Requests serialize and superseded requests cannot publish. Destroy before closing the explorer.
 */
export class GPUKeplerTableInspector {
  private readonly getDerivedColumns?: () => KeplerTableDerivedSnapshot | undefined;
  readonly maximumPageSize: number;
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly session: KeplerComputeSession;
  private readonly explorer: KeplerSessionExplorer;
  private readonly sortMemoryBudgetBytes: number;
  private sortProgram?: SortProgram;
  private requestRevision = 0;
  private readTail: Promise<void> = Promise.resolve();
  private destruction?: Promise<void>;
  private destroyed = false;
  private transferredBytes = 0;

  constructor(props: {
    session: KeplerComputeSession;
    explorer: KeplerSessionExplorer;
    /** Admission ceiling for the cached sort graph and owned permutations, reserved before compilation. */
    sortMemoryBudgetBytes: number;
    getDerivedColumns?: () => KeplerTableDerivedSnapshot | undefined;
  }) {
    if (!Number.isSafeInteger(props.sortMemoryBudgetBytes) || props.sortMemoryBudgetBytes < 0)
      throw new Error('Invalid table sort memory budget');
    this.session = props.session;
    this.explorer = props.explorer;
    this.sortMemoryBudgetBytes = props.sortMemoryBudgetBytes;
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(
      props.explorer.dataset.id,
      {allowIncomplete: props.explorer.allowIncompletePopulation}
    );
    if (this.lease.dataset.revision !== props.explorer.dataset.revision) {
      void this.lease.release();
      throw new Error('Table inspector and explorer require the same source revision');
    }
    this.getDerivedColumns = props.getDerivedColumns;
    this.maximumPageSize = props.explorer.maximumInspectionPageSize;
  }

  get columns(): readonly string[] {
    return [
      ...Object.keys(this.lease.dataset.value.columns),
      ...Object.keys(this.getDerivedColumns?.()?.columns ?? {})
    ];
  }

  /** Extra bounded permutation and derived reads; the explorer accounts for value/validity/page reads itself. */
  get readbackByteLength(): number {
    return this.transferredBytes;
  }

  async readPage(request: KeplerTablePageRequest): Promise<KeplerTablePage | null> {
    const derived = this.getDerivedColumns?.();
    const source = this.lease.dataset.value;
    if (derived && Object.keys(derived.columns).some(name => name in source.columns))
      throw new Error('Derived table fields must have distinct names');
    validateTablePageRequest(
      request,
      [...Object.keys(source.columns), ...Object.keys(derived?.columns ?? {})],
      this.maximumPageSize
    );
    const validity = {...source.analytics.validity, ...derived?.validity};
    const valuesPerRow =
      2 +
      request.columns.length +
      request.columns.filter(name => validity[name]).length +
      (request.sort ? 1 : 0);
    // Include selection counts and permutation IDs in the same explicit inspection budget.
    if (request.limit * valuesPerRow * 4 + 8 > this.explorer.maximumInspectionReadbackBytes)
      throw new KeplerAllocationError('memory-budget', {
        message: 'Table page exceeds readback byte budget'
      });
    // Capture mutable caller input before queuing the operation.
    const captured = {
      ...request,
      columns: [...request.columns],
      sort: request.sort ? {...request.sort} : undefined
    };
    const requestRevision = ++this.requestRevision;
    const result = this.readTail.then(() =>
      this.readCurrentPage(captured, requestRevision, derived)
    );
    this.readTail = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }

  /**
   * Read explicit source rows in the caller's order, such as a page of an exact CPU-ordered sort.
   * Uses the same bounds, serialization and revision gates as readPage; totalCount is the row count.
   */
  async readRows(
    sourceRows: readonly number[],
    columns: readonly string[]
  ): Promise<KeplerTablePage | null> {
    const derived = this.getDerivedColumns?.();
    const source = this.lease.dataset.value;
    if (sourceRows.length > this.maximumPageSize) throw new Error('Invalid bounded table page');
    const known = [...Object.keys(source.columns), ...Object.keys(derived?.columns ?? {})];
    if (new Set(columns).size !== columns.length || columns.some(name => !known.includes(name)))
      throw new Error('Unknown or duplicate table column');
    const requestRevision = ++this.requestRevision;
    const captured = [...sourceRows];
    const names = [...columns];
    const result = this.readTail.then(async () => {
      if (!this.isCurrent(requestRevision) || (derived && !derived.isCurrent())) return null;
      const selection = await this.explorer.readSelectedCount();
      if (!selection || !this.isCurrent(requestRevision, selection.revision)) return null;
      const page = await this.explorer.readRows(
        captured,
        names.filter(name => !derived?.columns[name])
      );
      const derivedNames = names.filter(name => derived?.columns[name]);
      if (page && derived && derivedNames.length) {
        const values = await this.readDerivedPage(page.sourceRowIndices, derivedNames, derived);
        Object.assign(page.columns, values.columns);
        Object.assign(page.validity, values.validity);
      }
      if (
        (derived && !derived.isCurrent()) ||
        !page ||
        page.revision !== selection.revision ||
        !this.isCurrent(requestRevision, selection.revision)
      )
        return null;
      return {...page, totalCount: captured.length};
    });
    this.readTail = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }

  /**
   * Whether readColumnStatistics can reduce this column on the GPU. A false result carries the
   * stated reason so the caller can fall back to a bounded CPU path or refuse visibly.
   */
  planColumnStatistics(column: string): GPUColumnStatisticsRoute {
    const derived = this.getDerivedColumns?.();
    const vector = derived?.columns[column] ?? this.lease.dataset.value.columns[column];
    if (!vector) return {supported: false, reason: 'Unknown table column.'};
    return planGPUColumnStatistics({
      format: vector.format,
      rowCount: this.lease.dataset.value.source.numRows
    });
  }

  /**
   * Fused GPU column profile (count, nulls, range, mean, deviation, histogram, distinct estimate)
   * for the whole table or the current selection. Only a few hundred bytes return to the CPU.
   */
  async readColumnStatistics(request: {
    column: string;
    population: 'all' | 'selected';
    histogramBinCount?: number;
  }): Promise<(GPUColumnStatistics & {revision: number}) | null> {
    const derived = this.getDerivedColumns?.();
    const source = this.lease.dataset.value;
    const requestRevision = ++this.requestRevision;
    const result = this.readTail.then(async () => {
      if (!this.isCurrent(requestRevision) || (derived && !derived.isCurrent())) return null;
      const selection = await this.explorer.readSelectedCount();
      if (!selection || !this.isCurrent(requestRevision, selection.revision)) return null;
      const values = derived?.columns[request.column] ?? source.columns[request.column];
      if (!values) throw new Error('Unknown table column');
      const populationRows =
        request.population === 'selected' ? selection.selectedCount : source.source.numRows;
      if (populationRows === 0) {
        return {
          populationRows,
          count: 0,
          nullCount: 0,
          minimum: NaN,
          maximum: NaN,
          sum: 0,
          mean: NaN,
          standardDeviation: NaN,
          sampleVariance: NaN,
          distinctEstimate: 0,
          histogram: new Uint32Array(
            request.histogramBinCount ?? DEFAULT_GPU_COLUMN_STATISTICS_BINS
          ),
          revision: selection.revision
        };
      }
      const histogramBinCount = request.histogramBinCount ?? DEFAULT_GPU_COLUMN_STATISTICS_BINS;
      const reservation = this.session.registry.reserveMemory(
        getGPUColumnStatisticsByteLength(source.source.numRows, histogramBinCount),
        'gpu',
        'scratch'
      );
      try {
        const statistics = await readGPUColumnStatistics({
          device: this.session.device,
          values: values as GPUVector<'float32'>,
          validity:
            derived?.validity?.[request.column] ?? source.analytics.validity[request.column],
          selection: request.population === 'selected' ? this.explorer.selectionMask : undefined,
          populationRows,
          histogramBinCount,
          isCurrent: () =>
            this.isCurrent(requestRevision, selection.revision) && (!derived || derived.isCurrent())
        });
        this.transferredBytes += 8 + (11 + histogramBinCount) * 4;
        return statistics && {...statistics, revision: selection.revision};
      } finally {
        reservation.release();
      }
    });
    this.readTail = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }

  /** Invalidates pending publication without attempting to preempt submitted GPU work. */
  cancel(): void {
    this.requestRevision++;
  }

  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.cancel();
    this.destruction = (async () => {
      await this.readTail;
      try {
        await this.releaseSortProgram();
      } finally {
        await this.lease.release();
      }
    })();
    return this.destruction;
  }

  private isCurrent(requestRevision: number, selectionRevision?: number): boolean {
    return (
      !this.destroyed &&
      requestRevision === this.requestRevision &&
      this.lease.isValid() &&
      this.explorer.isCurrent &&
      (selectionRevision === undefined || selectionRevision === this.explorer.currentRevision)
    );
  }

  private async readCurrentPage(
    request: KeplerTablePageRequest,
    requestRevision: number,
    derived?: KeplerTableDerivedSnapshot
  ): Promise<KeplerTablePage | null> {
    if (!this.isCurrent(requestRevision) || (derived && !derived.isCurrent())) return null;
    const selection = await this.explorer.readSelectedCount();
    if (!selection || !this.isCurrent(requestRevision, selection.revision)) return null;
    const revision = selection.revision;
    const totalCount =
      request.population === 'selected'
        ? selection.selectedCount
        : this.lease.dataset.value.source.numRows;
    const length = Math.min(request.limit, Math.max(0, totalCount - request.offset));
    const sourceColumns = request.columns.filter(name => !derived?.columns[name]);
    let page: GPUResidentPage | null;
    if (!request.sort || length === 0) {
      page =
        request.population === 'selected'
          ? await this.explorer.readPage(request.offset, request.limit, sourceColumns)
          : await this.explorer.readRows(
              Array.from({length}, (_, index) => request.offset + index),
              sourceColumns
            );
    } else {
      const program = await this.getSortProgram(request, derived);
      if (!this.isCurrent(requestRevision, revision) || (derived && !derived.isCurrent()))
        return null;
      if (program.revision !== revision) {
        const encoder = this.session.device.createCommandEncoder();
        program.query.encode(encoder);
        this.session.device.submit(encoder.finish());
        program.revision = revision;
      }
      const indices = program.query.globalRowIndices.data[0];
      this.transferredBytes += length * 4;
      const bytes = await getBuffer(indices).readAsync(
        indices.byteOffset + request.offset * 4,
        length * 4
      );
      if (!this.isCurrent(requestRevision, revision)) return null;
      const sourceRows = new Uint32Array(bytes.buffer, bytes.byteOffset, length);
      page = await this.explorer.readRows(Array.from(sourceRows), sourceColumns);
    }
    if (page && derived) {
      const names = request.columns.filter(name => derived.columns[name]);
      if (names.length) {
        if (!derived.isCurrent()) return null;
        const values = await this.readDerivedPage(page.sourceRowIndices, names, derived);
        Object.assign(page.columns, values.columns);
        Object.assign(page.validity, values.validity);
      }
    }
    if (
      (derived && !derived.isCurrent()) ||
      !page ||
      page.revision !== revision ||
      !this.isCurrent(requestRevision, revision)
    )
      return null;
    return {...page, totalCount};
  }

  private async getSortProgram(
    request: KeplerTablePageRequest,
    derived?: KeplerTableDerivedSnapshot
  ): Promise<SortProgram> {
    const sort = request.sort!;
    const key = JSON.stringify([
      sort.column,
      sort.direction,
      request.population,
      derived?.revision
    ]);
    if (this.sortProgram?.key === key) return this.sortProgram;
    await this.releaseSortProgram();
    const reservation = this.session.registry.reserveMemory(
      this.sortMemoryBudgetBytes,
      'gpu',
      'scratch'
    );
    let frame: GPUDataFrame<SortColumns> | undefined;
    let query: CompiledGPUDataFrameGlobalSort<SortColumns> | undefined;
    try {
      const source = this.lease.dataset.value;
      const validity = derived?.validity?.[sort.column] ?? source.analytics.validity[sort.column];
      const table = new GPUTable<SortColumns>({
        vectors: {
          value: borrowNamedVector(
            'value',
            derived?.columns[sort.column] ?? source.columns[sort.column]
          ),
          selected: borrowNamedVector('selected', this.explorer.selectionMask)
        }
      });
      for (const schema of [table.schema, ...table.batches.map(batch => batch.schema)]) {
        const field = schema.fields.find(candidate => candidate.name === 'value');
        if (field) field.nullable = Boolean(validity);
      }
      frame = new GPUDataFrame<SortColumns>({
        table,
        ...(validity ? {validity: {value: validity}} : {})
      });
      const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(this.session.device, {
        id: 'kepler-table-sort'
      });
      query =
        request.population === 'selected'
          ? frame
              .filter(column('selected').equal(literal(1)))
              .sortByGlobal('value', {direction: sort.direction, nulls: 'last', nans: 'last'})
              .compile(graph)
          : frame
              .sortByGlobal('value', {direction: sort.direction, nulls: 'last', nans: 'last'})
              .compile(graph);
      const buffers = new Set<Buffer>();
      for (const vector of [
        query.selectionMask,
        query.rowIndices,
        query.selectedCounts,
        query.globalRowIndices,
        query.globalSelectedCount
      ]) {
        for (const data of vector.data) buffers.add(getBuffer(data));
      }
      // Published compiled queries keep their graph private, so query scratch is not counted here:
      // this admission is a lower bound on real GPU memory (owned buffers only).
      const accountedBytes = [...buffers].reduce((sum, buffer) => sum + buffer.byteLength, 0);
      if (accountedBytes > reservation.byteLength)
        throw new KeplerAllocationError('memory-budget', {
          message: 'Table sort exceeds its reserved working memory'
        });
      this.sortProgram = {key, query, reservation, revision: -1};
      return this.sortProgram;
    } catch (error) {
      query?.destroy();
      reservation.release();
      throw error;
    } finally {
      frame?.destroy();
    }
  }

  private async readDerivedPage(
    rows: Uint32Array,
    names: readonly string[],
    snapshot: KeplerTableDerivedSnapshot
  ): Promise<Pick<GPUResidentPage, 'columns' | 'validity'>> {
    const result: Pick<GPUResidentPage, 'columns' | 'validity'> = {columns: {}, validity: {}};
    for (const name of names)
      result.columns[name] = decodeValues(snapshot.columns[name].format!, new Uint8Array());
    if (!rows.length) return result;
    const fieldCount = names.length + names.filter(name => snapshot.validity?.[name]).length;
    const reservation = this.session.registry.reserveMemory(
      (fieldCount + 1) * rows.byteLength * 2,
      'gpu',
      'readback'
    );
    const buffers: Buffer[] = [];
    let compiled: CompiledGPUCommandGraph | undefined;
    try {
      const graph = new GPUCommandGraph(this.session.device, {id: 'kepler-table-derived-page'});
      const indices = this.session.device.createBuffer({data: rows, usage: Buffer.STORAGE});
      buffers.push(indices);
      const indicesView = graph.importGPUData(
        'rows',
        new GPUData({buffer: indices, format: 'uint32', length: rows.length, ownsBuffer: false})
      );
      const outputs: {
        name: string;
        validity: boolean;
        format: GPUResidentScalarFormat;
        buffer: Buffer;
      }[] = [];
      for (const name of names) {
        for (const [validity, vector] of [
          [false, snapshot.columns[name]],
          [true, snapshot.validity?.[name]]
        ] as const) {
          if (!vector) continue;
          const buffer = this.session.device.createBuffer({
            byteLength: rows.byteLength,
            usage: Buffer.STORAGE | Buffer.COPY_SRC
          });
          buffers.push(buffer);
          outputs.push({name, validity, format: vector.format!, buffer});
          graph.add(
            new GPUGather({
              id: `derived-${name}-${validity}`,
              source: graph.importGPUVector(`source-${name}-${validity}`, vector),
              indices: indicesView,
              output: graph.importGPUData(
                `page-${name}-${validity}`,
                new GPUData({buffer, format: vector.format, length: rows.length, ownsBuffer: false})
              )
            })
          );
        }
      }
      compiled = graph.compile();
      const encoder = this.session.device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      this.session.device.submit(encoder.finish());
      this.transferredBytes += outputs.length * rows.byteLength;
      const values = await readAllBuffers(
        outputs.map(output => () => output.buffer.readAsync(0, rows.byteLength))
      );
      for (const [index, output] of outputs.entries()) {
        if (output.validity)
          result.validity[output.name] = new Uint32Array(
            values[index].buffer,
            values[index].byteOffset,
            rows.length
          );
        else result.columns[output.name] = decodeValues(output.format, values[index]);
      }
      return result;
    } finally {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
      reservation.release();
    }
  }

  private async releaseSortProgram(): Promise<void> {
    const program = this.sortProgram;
    if (!program) return;
    this.sortProgram = undefined;
    const fence = this.session.device.createFence();
    try {
      await Promise.race([fence.signaled, this.session.device.lost]);
    } finally {
      fence.destroy();
      program.query.destroy();
      program.reservation.release();
    }
  }
}

export function validateTablePageRequest(
  request: KeplerTablePageRequest,
  columns: readonly string[],
  maximumPageSize: number
): void {
  if (
    !Number.isSafeInteger(request.offset) ||
    request.offset < 0 ||
    !Number.isInteger(request.limit) ||
    request.limit < 1 ||
    request.limit > maximumPageSize
  )
    throw new Error('Invalid bounded table page');
  if (request.population !== 'all' && request.population !== 'selected')
    throw new Error('Invalid table population');
  if (
    new Set(request.columns).size !== request.columns.length ||
    request.columns.some(name => !columns.includes(name))
  )
    throw new Error('Unknown or duplicate table column');
  if (
    request.sort &&
    (!columns.includes(request.sort.column) ||
      !['ascending', 'descending'].includes(request.sort.direction))
  )
    throw new Error('Invalid table sort');
}

function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}

function borrowNamedVector<Format extends GPUResidentScalarFormat>(
  name: string,
  vector: GPUVector<Format>
): GPUVector<Format> {
  return new GPUVector({
    name,
    type: 'data',
    format: vector.format!,
    data: vector.data,
    ownsData: false
  });
}

function decodeValues(
  format: GPUResidentScalarFormat,
  bytes: Uint8Array
): Float32Array | Int32Array | Uint32Array {
  if (format === 'float32')
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  if (format === 'sint32')
    return new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
