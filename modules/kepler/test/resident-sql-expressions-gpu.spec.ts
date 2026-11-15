// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {
  ResidentSqlGpuRuntime,
  type ResidentSqlExecution
} from '../../../examples/deck/kepler-compute/src/resident-sql-gpu-runtime';
import {formatResidentSqlFloat} from '../../../examples/deck/kepler-compute/src/resident-sql-expressions';
import {compareResidentSqlStrings} from '../../../examples/deck/kepler-compute/src/resident-sql-strings';

const NAMES = [' Bob ', 'alice', 'Carol', 'carol', null, 'dave smith', 'Zoë'];
const CITIES = ['Paris', 'Rome', 'Oslo', 'Zurich'];
const OTHERS = ['Rome', 'Oslo', 'Paris'];
const DAYS = ['2024-01-01', '2024-03-15', '2023-12-31', '2024-03-15 08:30:00'];
const ROW_COUNT = 24;
const rows = Array.from({length: ROW_COUNT}, (_, id) => ({
  id,
  name: NAMES[id % NAMES.length] as string | null,
  city: CITIES[id % CITIES.length],
  other: OTHERS[(id * 2) % OTHERS.length],
  day: DAYS[id % DAYS.length],
  amount: id % 5 === 4 ? null : Math.fround(id * 1.5 - 7),
  quantity: (id * 7) % 11,
  epoch: 1_700_000_000 + id * 40_000 - 90_000 * (id % 3)
}));

async function makeRuntime() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 256 * 1024 * 1024});
  const source = new arrow.Table({
    id: arrow.makeVector(new Uint32Array(rows.map(row => row.id))),
    name: arrow.vectorFromArray(rows.map(row => row.name)),
    city: arrow.vectorFromArray(rows.map(row => row.city)),
    other: arrow.vectorFromArray(rows.map(row => row.other)),
    day: arrow.vectorFromArray(rows.map(row => row.day)),
    amount: arrow.vectorFromArray(
      rows.map(row => row.amount),
      new arrow.Float32()
    ),
    quantity: arrow.makeVector(new Int32Array(rows.map(row => row.quantity))),
    epoch: arrow.makeVector(new Int32Array(rows.map(row => row.epoch)))
  });
  // Two source batches: derived columns and results must keep chunk boundaries.
  session.importArrowDataset({id: 'places', table: source.slice(0, 10).concat(source.slice(10))});
  return {session, runtime: new ResidentSqlGpuRuntime(session)};
}

async function readRows(execution: ResidentSqlExecution): Promise<Record<string, unknown>[]> {
  const resident = execution.dataset.value.resident;
  if (resident)
    await resident.ensureRows(Array.from({length: execution.rowCount}, (_, index) => index));
  return execution.dataset.value.source.toArray().map(row => row.toJSON());
}

async function run(runtime: ResidentSqlGpuRuntime, sql: string) {
  const execution = await runtime.execute(runtime.planner.query(sql));
  return {execution, result: await readRows(execution)};
}

const trim = (text: string) => text.replace(/^ +| +$/g, '');

test('BETWEEN over numbers and strings matches an oracle', async () => {
  const {runtime} = await makeRuntime();
  const numeric = await run(runtime, 'SELECT id FROM places WHERE quantity BETWEEN 3 AND 6');
  expect(numeric.result.map(row => row['id'])).toEqual(
    rows.filter(row => row.quantity >= 3 && row.quantity <= 6).map(row => row.id)
  );
  const negated = await run(
    runtime,
    'SELECT id FROM places WHERE quantity + 1 NOT BETWEEN 3 AND 6 AND id < 12'
  );
  expect(negated.result.map(row => row['id'])).toEqual(
    rows
      .filter(row => (row.quantity + 1 < 3 || row.quantity + 1 > 6) && row.id < 12)
      .map(row => row.id)
  );
  const text = await run(runtime, "SELECT id FROM places WHERE city BETWEEN 'Oslo' AND 'Rome'");
  expect(text.result.map(row => row['id'])).toEqual(
    rows
      .filter(
        row =>
          compareResidentSqlStrings(row.city, 'Oslo') >= 0 &&
          compareResidentSqlStrings(row.city, 'Rome') <= 0
      )
      .map(row => row.id)
  );
  expect(numeric.execution.materialization).toBe('resident');
});

test('string functions run per label and gather by code on the GPU', async () => {
  const {runtime} = await makeRuntime();
  const named = rows.filter(row => row.name !== null);
  const upper = await run(
    runtime,
    'SELECT id, UPPER(name) AS shout FROM places WHERE name IS NOT NULL'
  );
  expect(upper.result.map(row => row['shout'])).toEqual(named.map(row => row.name!.toUpperCase()));
  const lengths = await run(
    runtime,
    'SELECT id, LENGTH(name) AS size FROM places WHERE LENGTH(name) > 3'
  );
  expect(lengths.result.map(row => [row['id'], Number(row['size'])])).toEqual(
    named
      .filter(row => Array.from(row.name!).length > 3)
      .map(row => [row.id, Array.from(row.name!).length])
  );
  const composed = await run(
    runtime,
    "SELECT id, REPLACE(SUBSTR(TRIM(name), 1, 4), 'a', '_') || '!' AS tag FROM places WHERE LOWER(name) = 'carol'"
  );
  expect(composed.result.map(row => row['tag'])).toEqual(
    named
      .filter(row => row.name!.toLowerCase() === 'carol')
      .map(row => `${Array.from(trim(row.name!)).slice(0, 4).join('').split('a').join('_')}!`)
  );
  const grouped = await run(
    runtime,
    'SELECT LOWER(name) AS key, COUNT(*) AS rows FROM places WHERE name IS NOT NULL GROUP BY LOWER(name)'
  );
  const expected = new Map<string, number>();
  for (const row of named)
    expected.set(row.name!.toLowerCase(), (expected.get(row.name!.toLowerCase()) ?? 0) + 1);
  expect(new Map(grouped.result.map(row => [row['key'], Number(row['rows'])]))).toEqual(expected);
});

test('two string columns concatenate (bounded) and compare through a shared dictionary', async () => {
  const {runtime} = await makeRuntime();
  const route = await run(runtime, "SELECT id, city || '>' || other AS route FROM places");
  expect(route.result.map(row => row['route'])).toEqual(
    rows.map(row => `${row.city}>${row.other}`)
  );
  const equal = await run(runtime, 'SELECT id FROM places WHERE city = other');
  expect(equal.result.map(row => row['id'])).toEqual(
    rows.filter(row => row.city === row.other).map(row => row.id)
  );
  const less = await run(runtime, 'SELECT id FROM places WHERE city < other');
  expect(less.result.map(row => row['id'])).toEqual(
    rows.filter(row => compareResidentSqlStrings(row.city, row.other) < 0).map(row => row.id)
  );
});

test('CAST numeric to VARCHAR builds a dictionary from a GPU distinct pass', async () => {
  const {runtime} = await makeRuntime();
  const labels = await run(runtime, 'SELECT id, CAST(quantity AS VARCHAR) AS label FROM places');
  expect(labels.result.map(row => row['label'])).toEqual(rows.map(row => String(row.quantity)));
  const filtered = await run(
    runtime,
    "SELECT id FROM places WHERE CAST(quantity AS VARCHAR) = '7'"
  );
  expect(filtered.result.map(row => row['id'])).toEqual(
    rows.filter(row => row.quantity === 7).map(row => row.id)
  );
  const floats = await run(
    runtime,
    'SELECT id, CAST(amount AS VARCHAR) AS text FROM places WHERE amount IS NOT NULL'
  );
  expect(floats.result.map(row => row['text'])).toEqual(
    rows.filter(row => row.amount !== null).map(row => formatResidentSqlFloat(row.amount!))
  );
  const grouped = await run(
    runtime,
    'SELECT CAST(quantity AS VARCHAR) AS q, COUNT(*) AS rows FROM places GROUP BY CAST(quantity AS VARCHAR)'
  );
  const expected = new Map<string, number>();
  for (const row of rows)
    expected.set(String(row.quantity), (expected.get(String(row.quantity)) ?? 0) + 1);
  expect(new Map(grouped.result.map(row => [row['q'], Number(row['rows'])]))).toEqual(expected);
});

test('CAST of expressions and to DATE/TIMESTAMP use epoch-second math', async () => {
  const {runtime} = await makeRuntime();
  const expression = await run(
    runtime,
    'SELECT id, CAST(quantity * 3 - id AS INTEGER) AS mixed, CAST(amount / 2 AS INTEGER) AS half FROM places'
  );
  for (const row of rows) {
    const cells = expression.result[row.id];
    expect(Number(cells['mixed'])).toBe(row.quantity * 3 - row.id);
    if (row.amount === null) expect(cells['half']).toBeNull();
    else {
      const half = Math.fround(row.amount / 2);
      expect(Number(cells['half'])).toBe(Math.sign(half) * Math.floor(Math.abs(half) + 0.5));
    }
  }
  const dates = await run(
    runtime,
    'SELECT id, CAST(epoch AS TIMESTAMP) AS ts, CAST(epoch AS DATE) AS day_epoch, CAST(day AS DATE) AS parsed FROM places'
  );
  for (const row of rows) {
    const cells = dates.result[row.id];
    expect(Number(cells['ts'])).toBe(row.epoch);
    expect(Number(cells['day_epoch'])).toBe(Math.floor(row.epoch / 86400) * 86400);
    expect(Number(cells['parsed'])).toBe(
      Math.floor(Date.parse(`${row.day.slice(0, 10)}T00:00:00Z`) / 1000)
    );
  }
  const field = dates.execution.dataset.value.source.schema.fields.find(
    entry => entry.name === 'ts'
  );
  expect(field?.metadata.get('luma.gl.time.format')).toBe('X');
});

test('DATE and TIMESTAMP columns compare with string literals as epoch seconds', async () => {
  const {runtime} = await makeRuntime();
  const cutoff = Date.UTC(2023, 10, 15, 0, 0, 0) / 1000;
  const timestamps = await run(
    runtime,
    "SELECT id FROM places WHERE CAST(epoch AS TIMESTAMP) >= '2023-11-15 00:00:00'"
  );
  expect(timestamps.result.map(row => row['id'])).toEqual(
    rows.filter(row => row.epoch >= cutoff).map(row => row.id)
  );
  const between = await run(
    runtime,
    "SELECT id FROM places WHERE CAST(day AS DATE) BETWEEN '2024-01-01' AND '2024-03-15'"
  );
  expect(between.result.map(row => row['id'])).toEqual(
    rows
      .filter(row => row.day.slice(0, 10) >= '2024-01-01' && row.day.slice(0, 10) <= '2024-03-15')
      .map(row => row.id)
  );
});

test('string expressions over three and more columns use a bounded mixed-radix lookup', async () => {
  const {runtime} = await makeRuntime();
  const joined = await run(
    runtime,
    "SELECT id, city || '-' || other || '-' || name AS label FROM places"
  );
  expect(joined.result.map(row => row['label'] ?? null)).toEqual(
    rows.map(row => (row.name === null ? null : `${row.city}-${row.other}-${row.name}`))
  );
  const upper = await run(
    runtime,
    'SELECT UPPER(city || other || day) AS key, COUNT(*) AS rows FROM places GROUP BY UPPER(city || other || day)'
  );
  const expected = new Map<string, number>();
  for (const row of rows) {
    const key = `${row.city}${row.other}${row.day}`.toUpperCase();
    expected.set(key, (expected.get(key) ?? 0) + 1);
  }
  expect(new Map(upper.result.map(row => [row['key'], Number(row['rows'])]))).toEqual(expected);
  const four = await run(
    runtime,
    "SELECT id FROM places WHERE name || city || other || day <> 'x'"
  );
  expect(four.result.map(row => row['id'])).toEqual(
    rows.filter(row => row.name !== null).map(row => row.id)
  );
  // An empty result cannot be a resident dataset: a refusal that the router retries on DuckDB.
  await expect(
    run(runtime, "SELECT id FROM places WHERE name || city || other = 'x'")
  ).rejects.toMatchObject({name: 'SqlExecutionRefusal'});
});

test('CAST to DATE or TIMESTAMP publishes exact time words for the native time filter', async () => {
  const {runtime} = await makeRuntime();
  const {execution} = await run(
    runtime,
    'SELECT id, CAST(epoch AS TIMESTAMP) AS ts, CAST(quantity AS VARCHAR) AS q FROM places'
  );
  const time = execution.dataset.value.timeColumns['ts'];
  expect(time?.metadata.unit).toBe('second');
  expect(Object.keys(execution.dataset.value.timeColumns)).toEqual(['ts']);
  expect(execution.dataset.value.prepared.timeColumns['ts']).toBe(time.metadata);
  const words = execution.dataset.value.prepared.timeColumns['ts'].chunks.flatMap(chunk =>
    Array.from({length: chunk.length / 2}, (_, row) =>
      Number(BigInt.asIntN(64, BigInt(chunk[row * 2]) | (BigInt(chunk[row * 2 + 1]) << 32n)))
    )
  );
  expect(words).toEqual(rows.map(row => row.epoch));
  // The scalar column remains for SQL and display; the other columns are not time columns.
  expect(execution.dataset.value.columns['ts']).toBeDefined();
});

test('CAST numeric to VARCHAR lifts the old 65,536 distinct limit and refuses beyond its bound', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 512 * 1024 * 1024});
  const count = 70_000;
  session.importArrowDataset({
    id: 'wide',
    table: new arrow.Table({
      id: arrow.makeVector(Uint32Array.from({length: count}, (_, index) => index * 3))
    })
  });
  const runtime = new ResidentSqlGpuRuntime(session);
  const {result, execution} = await run(
    runtime,
    'SELECT id, CAST(id AS VARCHAR) AS label FROM wide'
  );
  expect(execution.rowCount).toBe(count);
  for (const row of result) expect(row['label']).toBe(String(row['id']));
  const {findResidentSqlDistinctKeys} = await import(
    '../../../examples/deck/kepler-compute/src/resident-sql-distinct'
  );
  const {SqlExecutionRefusal} = await import(
    '../../../examples/deck/kepler-compute/src/sql-backend-router'
  );
  const vector = session.registry.getDataset<{columns: Record<string, never>}>('wide')!.value
    .columns['id'];
  await expect(
    findResidentSqlDistinctKeys({device, vector, format: 'uint32', maximumCount: 100})
  ).rejects.toBeInstanceOf(SqlExecutionRefusal);
});
