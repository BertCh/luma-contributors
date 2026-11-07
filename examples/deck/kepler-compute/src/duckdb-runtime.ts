// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  createKeplerDuckDBExecutor,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetReference,
  type KeplerDatasetLease,
  type KeplerDatasetSnapshot,
  type KeplerDuckDBConnection,
  type KeplerDuckDBResult,
  type KeplerMemoryReservation
} from '@luma-contributors/kepler';

import {materializeResidentRows} from './resident-materialization';

export const DEFAULT_DUCKDB_SQL = `SELECT category, CAST(AVG(value) AS FLOAT) AS value,
  CAST(COUNT(*) AS INTEGER) AS count
FROM observations GROUP BY category ORDER BY category`;

export type OwnedDuckDBConnection = KeplerDuckDBConnection & {destroy(): Promise<void>};
export type DuckDBConnectionFactory = () => Promise<OwnedDuckDBConnection>;
export type DuckDBRuntimeStatus = 'disabled' | 'starting' | 'ready' | 'busy' | 'closed';
/** Borrowed CPU Arrow result. Revalidate this identity before publishing an asynchronous preview. */
export type DuckDBResultSnapshot = {
  readonly generation: number;
  readonly source: Readonly<KeplerDatasetReference>;
  readonly table: arrow.Table;
  readonly sql: string;
};

/** Optional CPU SQL complement. Every Arrow exchange and GPU upload is an explicit operation. */
export class DuckDBRuntime {
  private readonly session: KeplerComputeSession;
  private readonly source: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly factory: DuckDBConnectionFactory;
  private connection?: OwnedDuckDBConnection;
  private executor?: ReturnType<typeof createKeplerDuckDBExecutor>;
  private result?: KeplerDuckDBResult;
  private resultSnapshot?: DuckDBResultSnapshot;
  private generation = 0;
  private resultReservation?: KeplerMemoryReservation;
  private imported?: KeplerDatasetSnapshot<KeplerArrowDataset>;
  private pending?: Promise<unknown>;
  private listeners = new Set<() => void>();
  private closed = false;
  private destruction?: Promise<void>;
  private currentStatus: DuckDBRuntimeStatus = 'disabled';
  private currentError = '';
  private registered = false;
  /** Extra relations (other resident datasets, alternate names) by name to the revision registered. */
  private readonly extraRelations = new Map<string, number>();
  private sourceArrowBytes = 0;
  private resultArrowBytes = 0;
  private uploadedBytes = 0;
  private uploads = 0;

  constructor(props: {
    session: KeplerComputeSession;
    datasetId: string;
    createConnection?: DuckDBConnectionFactory;
  }) {
    this.session = props.session;
    this.source = props.session.registry.acquireDataset<KeplerArrowDataset>(props.datasetId);
    this.factory = props.createConnection ?? createBrowserDuckDBConnection;
  }

  get status(): DuckDBRuntimeStatus {
    return this.currentStatus;
  }
  get error(): string {
    return this.currentError;
  }
  get sourceRegistered(): boolean {
    return this.registered;
  }
  get sourceRevision(): number {
    return this.source.dataset.revision;
  }
  get resultInfo() {
    return (
      this.currentResult &&
      this.result && {
        generation: this.generation,
        rowCount: this.result.rowCount,
        columns: this.result.columns
      }
    );
  }
  /** Reading CPU Arrow never normalizes or uploads it. */
  get resultArrowTable(): arrow.Table | undefined {
    return this.currentResult?.table;
  }
  get resultGeneration(): number {
    return this.generation;
  }
  get currentResult(): DuckDBResultSnapshot | undefined {
    return this.resultSnapshot && this.isResultCurrent(this.resultSnapshot)
      ? this.resultSnapshot
      : undefined;
  }
  isResultCurrent(result: DuckDBResultSnapshot): boolean {
    return !this.pending && this.isSourceCurrent() && this.resultSnapshot === result;
  }
  get importedDataset() {
    return this.currentResult ? this.imported : undefined;
  }
  get statistics() {
    return {
      ...this.executor?.statistics,
      sourceArrowBytes: this.sourceArrowBytes,
      resultArrowBytes: this.resultArrowBytes,
      uploadedBytes: this.uploadedBytes,
      uploads: this.uploads,
      retainedResultBytes: this.resultReservation?.byteLength ?? 0
    };
  }
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  initialize(): Promise<void> {
    return this.perform(async () => {
      if (this.connection) return;
      this.currentStatus = 'starting';
      this.notify();
      const connection = await this.factory();
      if (this.closed) {
        await connection.destroy();
        return;
      }
      this.connection = connection;
      this.executor = createKeplerDuckDBExecutor(connection);
    });
  }

  /** Registers retained CPU Arrow, never reads GPU buffers or automatically mirrors filters. */
  registerSource(): Promise<void> {
    return this.perform(async () => {
      this.assertSourceCurrent();
      if (!this.executor) throw new Error('Enable DuckDB first');
      if (this.registered) return;
      const source = this.source.dataset;
      // Resident results stream GPU pages into the accounted lazy Arrow view, batch by batch,
      // before DuckDB receives the real record batches.
      await materializeResidentRows(source.value, undefined, {
        assertCurrent: () => this.assertSourceCurrent()
      });
      const transferBytes = arrow.tableToIPC(source.value.source, 'stream').byteLength;
      await this.executor.registerArrowResult('observations', source.value.source, {
        datasetId: source.id,
        revision: source.revision
      });
      this.sourceArrowBytes += transferBytes;
      this.registered = true;
    });
  }

  /**
   * Registers every resident dataset the statement names (by id or by its bare identifier form),
   * so joins and fallbacks over several resident tables see the same names as the GPU planner.
   * Each relation is registered once per name; a dataset that changed afterwards must be renamed.
   */
  private async registerReferencedRelations(sql: string): Promise<void> {
    const executor = this.executor!;
    for (const {id} of this.session.registry.getDatasetFootprints()) {
      const bareName = id.replace(/[^A-Za-z0-9_]/g, '_');
      for (const name of new Set([id, bareName])) {
        if (name === 'observations') continue;
        const escaped = name.replace(/[.*+?^${}()|[\]\\-]/g, '\\$&');
        const pattern = /^[A-Za-z0-9_]+$/.test(name)
          ? new RegExp(`(?<![A-Za-z0-9_"])${escaped}(?![A-Za-z0-9_"])|"${escaped}"`, 'i')
          : new RegExp(`"${escaped}"`, 'i');
        if (!pattern.test(sql)) continue;
        const dataset = this.session.registry.getDataset<KeplerArrowDataset>(id);
        if (!dataset) continue;
        const registeredRevision = this.extraRelations.get(name);
        if (registeredRevision === dataset.revision) continue;
        if (registeredRevision !== undefined)
          throw new Error(`Table ${name} changed after DuckDB registered it; reload the explorer`);
        await materializeResidentRows(dataset.value, undefined, {
          assertCurrent: () => {
            if (this.session.registry.getDataset(id)?.revision !== dataset.revision)
              throw new Error(`DuckDB source ${id} is stale or unavailable`);
          }
        });
        this.sourceArrowBytes += arrow.tableToIPC(dataset.value.source, 'stream').byteLength;
        await executor.registerArrowResult(name, dataset.value.source, {
          datasetId: id,
          revision: dataset.revision
        });
        this.extraRelations.set(name, dataset.revision);
      }
    }
  }

  runSql(sql: string): Promise<void> {
    return this.perform(async () => {
      this.assertSourceCurrent();
      if (!this.executor || !this.registered)
        throw new Error('Register the source Arrow table first');
      await this.registerReferencedRelations(sql);
      if (this.closed) return;
      this.assertSourceCurrent();
      const result = await this.executor.execute({kind: 'sql', sql});
      if (this.closed) return;
      this.assertSourceCurrent();
      const retainedBytes = getRetainedArrowBytes(result.table);
      const transferBytes = arrow.tableToIPC(result.table, 'stream').byteLength;
      const reservation = this.session.registry.reserveMemory(retainedBytes, 'cpu', 'output');
      this.resultReservation?.release();
      this.resultReservation = reservation;
      this.result = result;
      this.resultSnapshot = Object.freeze({
        generation: this.generation,
        source: Object.freeze({
          datasetId: this.source.dataset.id,
          revision: this.source.dataset.revision
        }),
        table: result.table,
        sql
      });
      this.resultArrowBytes += transferBytes;
    }, true);
  }

  /** SQL stays CPU Arrow until this explicit normalization/upload. Full result upload is visible. */
  importResult(expectedGeneration = this.generation): KeplerDatasetSnapshot<KeplerArrowDataset> {
    this.assertSourceCurrent();
    if (this.pending) throw new Error('Wait for the current SQL operation');
    if (!this.result) throw new Error('Run SQL before importing its result');
    if (expectedGeneration !== this.generation) throw new Error('SQL result generation is stale');
    if (this.imported && this.session.registry.getDataset(this.imported.id) === this.imported)
      return this.imported;
    const dataset = this.session.importArrowDataset({
      id: `${this.source.dataset.id}-sql-result`,
      table: this.result.table
    });
    this.imported = dataset;
    this.uploadedBytes += dataset.value.uploadedBufferBytes;
    this.uploads++;
    // The resident dataset now accounts for the same retained Arrow buffers.
    this.resultReservation?.release();
    this.resultReservation = undefined;
    this.notify();
    return dataset;
  }

  /** A bounded CPU preview of an explicit SQL result; no GPU readback occurs here. */
  previewRows(limit = 8): readonly Record<string, unknown>[] {
    const table = this.resultArrowTable;
    if (!table) return [];
    return Array.from({length: Math.min(Math.max(0, limit), 16, table.numRows)}, (_, index) => {
      const row: Record<string, unknown> = {};
      for (const field of table.schema.fields.slice(0, 8))
        row[field.name] = table.getChild(field.name)?.get(index);
      return row;
    });
  }

  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.closed = true;
    this.generation++;
    this.resultSnapshot = undefined;
    this.currentStatus = 'closed';
    this.listeners.clear();
    this.destruction = (async () => {
      await this.pending?.catch(() => {});
      try {
        await this.connection?.destroy();
      } finally {
        this.resultReservation?.release();
        this.resultReservation = undefined;
        this.result = undefined;
        await this.source.release();
      }
    })();
    return this.destruction;
  }

  private perform(operation: () => Promise<void>, replaceResult = false): Promise<void> {
    if (this.closed) return Promise.reject(new Error('DuckDB runtime is closed'));
    if (this.pending) return Promise.reject(new Error('DuckDB operation already running'));
    this.currentStatus = 'busy';
    this.currentError = '';
    if (replaceResult) {
      this.generation++;
      this.resultSnapshot = undefined;
      this.result = undefined;
      this.imported = undefined;
      this.resultReservation?.release();
      this.resultReservation = undefined;
    }
    const pending = Promise.resolve()
      .then(operation)
      .catch(error => {
        if (!this.closed)
          this.currentError = error instanceof Error ? error.message : String(error);
        throw error;
      })
      .finally(() => {
        this.pending = undefined;
        if (!this.closed) this.currentStatus = this.connection ? 'ready' : 'disabled';
        this.notify();
      });
    this.pending = pending;
    this.notify();
    return pending;
  }
  private assertSourceCurrent(): void {
    if (!this.isSourceCurrent()) throw new Error('DuckDB source is stale or unavailable');
  }
  private isSourceCurrent(): boolean {
    return (
      !this.closed &&
      this.source.isValid() &&
      this.session.registry.getDataset(this.source.dataset.id)?.revision ===
        this.source.dataset.revision
    );
  }
  private notify(): void {
    for (const listener of this.listeners) listener();
  }
}

/** Lazy local assets: opening the map does not instantiate or download DuckDB-WASM. */
export async function createBrowserDuckDBConnection(): Promise<OwnedDuckDBConnection> {
  const {createConnection} = await import('./duckdb-browser');
  return createConnection();
}

function getRetainedArrowBytes(table: arrow.Table): number {
  const buffers = new Set<ArrayBufferLike>();
  const visit = (data: arrow.Data) => {
    for (const buffer of [data.values, data.valueOffsets, data.nullBitmap, data.typeIds])
      if (buffer) buffers.add(buffer.buffer);
    for (const child of data.children) visit(child);
    for (const dictionary of data.dictionary?.data ?? []) visit(dictionary);
  };
  for (const batch of table.batches) visit(batch.data);
  return [...buffers].reduce((total, buffer) => total + buffer.byteLength, 0);
}
