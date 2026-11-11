// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';
import {KeplerComputeSession} from '../src/session/compute-session';
import {DuckDBRuntime} from '../../../examples/deck/kepler-compute/src/duckdb-runtime';

const makeTable = (value: number) => arrow.tableFromArrays({value: new Float32Array([value])});

// Promise.withResolvers is ES2024; the test typecheck targets ES2022 lib.
function createDeferredTable() {
  let resolve!: (table: arrow.Table) => void;
  const promise = new Promise<arrow.Table>(promiseResolve => {
    resolve = promiseResolve;
  });
  return {promise, resolve};
}

async function createFixture() {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  session.importArrowDataset({id: 'source', table: makeTable(1)});
  const connection = {
    query: vi.fn(async (_sql: string) => makeTable(2)),
    insertArrowTable: vi.fn(async () => {}),
    destroy: vi.fn(async () => {})
  };
  const runtime = new DuckDBRuntime({
    session,
    datasetId: 'source',
    createConnection: async () => connection
  });
  await runtime.initialize();
  await runtime.registerSource();
  return {
    device,
    session,
    connection,
    runtime,
    async destroy() {
      await runtime.destroy();
      await session.destroy(Promise.resolve());
      device.destroy();
    }
  };
}

test('SQL preview exposes a stable borrowed generation without upload and import reuses its resident result', async () => {
  const fixture = await createFixture();
  const {runtime, device} = fixture;
  const createBuffer = vi.spyOn(device, 'createBuffer');
  try {
    await runtime.runSql('SELECT value FROM observations');
    const result = runtime.currentResult!;
    expect(result.generation).toBe(1);
    expect(result.source).toEqual({datasetId: 'source', revision: 1});
    expect(result.sql).toBe('SELECT value FROM observations');
    expect(result.table).toBe(runtime.resultArrowTable);
    expect(Object.isFrozen(result)).toBe(true);
    expect(runtime.currentResult).toBe(result);
    expect(runtime.isResultCurrent(result)).toBe(true);
    expect(createBuffer).not.toHaveBeenCalled();
    expect(runtime.statistics.uploads).toBe(0);
    const resident = runtime.importResult(result.generation);
    const allocations = createBuffer.mock.calls.length;
    expect(runtime.importResult(result.generation)).toBe(resident);
    expect(runtime.importedDataset).toBe(resident);
    expect(createBuffer).toHaveBeenCalledTimes(allocations);
    expect(runtime.statistics.uploads).toBe(1);
  } finally {
    createBuffer.mockRestore();
    await fixture.destroy();
  }
});

test('a newer query immediately invalidates prepared generations and failed queries do not revive old results', async () => {
  const fixture = await createFixture();
  const {runtime, connection} = fixture;
  try {
    await runtime.runSql('first');
    const first = runtime.currentResult!;
    const next = createDeferredTable();
    connection.query.mockReturnValueOnce(next.promise);
    const pending = runtime.runSql('second');
    expect(runtime.isResultCurrent(first)).toBe(false);
    expect(runtime.currentResult).toBeUndefined();
    expect(runtime.resultArrowTable).toBeUndefined();
    expect(runtime.resultGeneration).toBe(2);
    expect(runtime.statistics.retainedResultBytes).toBe(0);
    await expect(runtime.runSql('overlapping')).rejects.toThrow('already running');
    expect(runtime.resultGeneration).toBe(2);
    next.resolve(makeTable(3));
    await pending;
    expect(runtime.currentResult!.generation).toBe(2);
    expect(runtime.currentResult!.table.getChild('value')!.get(0)).toBe(3);
    expect(() => runtime.importResult(first.generation)).toThrow('generation is stale');
    connection.query.mockRejectedValueOnce(new Error('invalid SQL'));
    await expect(runtime.runSql('bad')).rejects.toThrow('invalid SQL');
    expect(runtime.resultArrowTable).toBeUndefined();
    expect(runtime.currentResult).toBeUndefined();
    expect(runtime.statistics.uploads).toBe(0);
  } finally {
    await fixture.destroy();
  }
});

test('source replacement and runtime destruction invalidate asynchronous publication identities', async () => {
  const fixture = await createFixture();
  const {runtime, session, connection} = fixture;
  try {
    await runtime.runSql('first');
    const result = runtime.currentResult!;
    session.importArrowDataset({id: 'source', table: makeTable(9)});
    expect(runtime.isResultCurrent(result)).toBe(false);
    expect(runtime.resultArrowTable).toBeUndefined();
    expect(() => runtime.importResult(result.generation)).toThrow('stale');
  } finally {
    await fixture.destroy();
  }
  expect(connection.destroy).toHaveBeenCalledOnce();

  const pendingFixture = await createFixture();
  try {
    const next = createDeferredTable();
    pendingFixture.connection.query.mockReturnValueOnce(next.promise);
    const query = pendingFixture.runtime.runSql('pending');
    await Promise.resolve();
    const destruction = pendingFixture.runtime.destroy();
    next.resolve(makeTable(10));
    await query;
    await destruction;
    expect(pendingFixture.runtime.currentResult).toBeUndefined();
    expect(pendingFixture.runtime.resultArrowTable).toBeUndefined();
    expect(pendingFixture.runtime.statistics.retainedResultBytes).toBe(0);
  } finally {
    await pendingFixture.destroy();
  }
});
