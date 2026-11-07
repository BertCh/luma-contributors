// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as duckdb from '@duckdb/duckdb-wasm';
import * as arrow from 'apache-arrow';
import mainModule from '@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url';
import workerUrl from '@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url';
import type {OwnedDuckDBConnection} from './duckdb-runtime';

export async function createConnection(): Promise<OwnedDuckDBConnection> {
  const worker = new Worker(workerUrl);
  const database = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
  try {
    await database.instantiate(mainModule);
    await database.open({});
    const connection = await database.connect();
    return {
      query: sql => connection.query(sql),
      insertArrowTable: (table, options) =>
        connection.insertArrowFromIPCStream(arrow.tableToIPC(table, 'stream'), {
          ...options,
          create: true
        }),
      async destroy() {
        try {
          await connection.close();
        } finally {
          await database.terminate();
          worker.terminate();
        }
      }
    };
  } catch (error) {
    await database.terminate();
    worker.terminate();
    throw error;
  }
}
