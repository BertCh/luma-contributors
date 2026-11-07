// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import * as duckdb from '@duckdb/duckdb-wasm/blocking';
import {createKeplerDuckDBExecutor} from '../src/execution/duckdb-executor';

it('the optional real DuckDB-WASM complement registers Arrow and returns typed grouped Arrow', async () => {
  const require = createRequire(import.meta.url);
  const database = await duckdb.createDuckDB(
    {
      mvp: {
        mainModule: require.resolve('@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm')
      },
      eh: {
        mainModule: require.resolve('@duckdb/duckdb-wasm/dist/duckdb-eh.wasm')
      }
    },
    new duckdb.VoidLogger(),
    duckdb.NODE_RUNTIME
  );
  await database.instantiate();
  database.open({});
  const connection = database.connect();
  try {
    const executor = createKeplerDuckDBExecutor({
      async query(sql) {
        return connection.query(sql);
      },
      async insertArrowTable(table, options) {
        connection.insertArrowFromIPCStream(arrow.tableToIPC(table, 'stream'), {
          ...options,
          create: true
        });
      }
    });
    await executor.registerArrowResult(
      'observations',
      arrow.tableFromArrays({
        category: new Uint32Array([0, 1, 0, 1]),
        value: new Float32Array([1, 10, 3, 20])
      }),
      {datasetId: 'observations', revision: 1}
    );
    const result = await executor.execute({
      kind: 'sql',
      sql: `SELECT category,
      CAST(AVG(value) AS FLOAT) AS value, CAST(COUNT(*) AS INTEGER) AS count
      FROM observations GROUP BY category ORDER BY category`
    });
    expect(result.residency).toBe('cpu-arrow');
    expect(result.rowCount).toBe(2);
    expect(result.columns).toEqual(['category', 'value', 'count']);
    expect(Array.from(result.table.getChild('value')!.toArray())).toEqual([2, 15]);
    expect(Array.from(result.table.getChild('count')!.toArray())).toEqual([2, 2]);
    expect(result.table.getChild('value')!.type.toString()).toBe('Float32');
    expect(executor.statistics).toEqual({
      queries: 1,
      importedTables: 1,
      importedRows: 4,
      resultRows: 2
    });
    expect(executor.getRelationSource('observations')).toEqual({
      datasetId: 'observations',
      revision: 1
    });
  } finally {
    try {
      connection.close();
    } finally {
      database.reset();
    }
  }
}, 30000);
