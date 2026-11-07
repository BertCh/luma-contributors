// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Table} from 'apache-arrow';
import type {KeplerDatasetReference} from '../recipes/analysis-recipe';

/** A caller-owned DuckDB-WASM connection; this adapter imports no DuckDB runtime. */
export type KeplerDuckDBConnection = {
  query(sql: string): Promise<Table>;
  insertArrowTable(table: Table, options: {name: string}): Promise<void>;
};
export type KeplerDuckDBResult = {
  backend: 'duckdb';
  table: Table;
  rowCount: number;
  columns: readonly string[];
  /** No GPU upload has occurred. The host explicitly normalizes/uploads desired columns. */
  residency: 'cpu-arrow';
};

/** Explicit Arrow exchanges and counters; connection lifetime remains with the application. */
export function createKeplerDuckDBExecutor(connection: KeplerDuckDBConnection) {
  const statistics = {queries: 0, importedTables: 0, importedRows: 0, resultRows: 0};
  const relations = new Map<string, KeplerDatasetReference>();
  const pendingRelations = new Set<string>();
  return {
    get statistics() {
      return {...statistics};
    },
    getRelationSource(name: string) {
      const source = relations.get(name);
      return source && {...source};
    },
    async execute(
      request: {kind: 'sql'; sql: string},
      signal?: AbortSignal
    ): Promise<KeplerDuckDBResult> {
      signal?.throwIfAborted();
      if (!request.sql.trim()) throw new Error('SQL must not be empty');
      statistics.queries++;
      const table = await connection.query(request.sql);
      statistics.resultRows += table.numRows;
      signal?.throwIfAborted();
      return {
        backend: 'duckdb',
        table,
        rowCount: table.numRows,
        columns: table.schema.fields.map(field => field.name),
        residency: 'cpu-arrow'
      };
    },
    /** Caller must materialize GPU output to Arrow explicitly before calling this method. */
    async registerArrowResult(
      name: string,
      table: Table,
      source: KeplerDatasetReference,
      signal?: AbortSignal
    ): Promise<void> {
      signal?.throwIfAborted();
      if (
        !name ||
        !source.datasetId ||
        !Number.isSafeInteger(source.revision) ||
        source.revision < 1
      )
        throw new Error('Invalid DuckDB relation source');
      // Refuse ambiguous overwrite; callers choose a new relation per result revision.
      if (relations.has(name) || pendingRelations.has(name))
        throw new Error('DuckDB relation name already registered');
      const sourceSnapshot = Object.freeze({...source});
      pendingRelations.add(name);
      try {
        await connection.insertArrowTable(table, {name});
        relations.set(name, sourceSnapshot);
        statistics.importedTables++;
        statistics.importedRows += table.numRows;
      } finally {
        pendingRelations.delete(name);
      }
      // Insertion is committed even if the request was cancelled while DuckDB worked.
      signal?.throwIfAborted();
    }
  };
}
