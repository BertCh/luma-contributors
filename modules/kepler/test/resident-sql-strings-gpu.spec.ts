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
import {
  castResidentSqlValue,
  compareResidentSqlStrings
} from '../../../examples/deck/kepler-compute/src/resident-sql-strings';

const POOL = ['Zurich', 'Alpha', 'Beta', 'alphabet', 'Be_ta', null, 'Zürich'];
const ROW_COUNT = 20;
const rows = Array.from({length: ROW_COUNT}, (_, id) => ({
  id,
  city: POOL[id % POOL.length] as string | null,
  amount: id % 4 === 3 ? null : Math.fround(id * 1.25 - 6),
  quantity: id * 3 - 20
}));

async function makeRuntime() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 256 * 1024 * 1024});
  const source = new arrow.Table({
    id: arrow.makeVector(new Uint32Array(rows.map(row => row.id))),
    city: arrow.vectorFromArray(rows.map(row => row.city)),
    amount: arrow.vectorFromArray(
      rows.map(row => row.amount),
      new arrow.Float32()
    ),
    quantity: arrow.makeVector(new Int32Array(rows.map(row => row.quantity)))
  });
  // Two source batches: derived columns and results must keep chunk boundaries.
  session.importArrowDataset({id: 'places', table: source.slice(0, 9).concat(source.slice(9))});
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

test('string predicates run on the GPU and match a JavaScript oracle', async () => {
  const {runtime} = await makeRuntime();
  const likePrefix = rows.filter(row => row.city?.startsWith('Be')).map(row => row.id);
  const cases: [string, (row: (typeof rows)[number]) => boolean][] = [
    ["city = 'Beta'", row => row.city === 'Beta'],
    ["city <> 'Beta'", row => row.city !== null && row.city !== 'Beta'],
    ["city IN ('Beta', 'Alpha', 'Nowhere')", row => ['Beta', 'Alpha'].includes(row.city ?? '')],
    ["city NOT IN ('Beta')", row => row.city !== null && row.city !== 'Beta'],
    ["city LIKE 'Be%'", row => Boolean(row.city?.startsWith('Be'))],
    ["city LIKE '%ta'", row => Boolean(row.city?.endsWith('ta'))],
    ["city LIKE '%lph%'", row => Boolean(row.city?.includes('lph'))],
    ["city ILIKE '%ALPH%'", row => Boolean(row.city?.toLowerCase().includes('alph'))],
    ["city LIKE 'Be_ta'", row => row.city === 'Be_ta'],
    ["city NOT LIKE 'Z%'", row => row.city !== null && !row.city.startsWith('Z')],
    [
      "city >= 'Beta'",
      row => row.city !== null && compareResidentSqlStrings(row.city, 'Beta') >= 0
    ],
    [
      "city LIKE 'Be%' AND amount > 0",
      row => Boolean(row.city?.startsWith('Be')) && (row.amount ?? -1) > 0
    ],
    ["city = 'Zurich' OR quantity > 30", row => row.city === 'Zurich' || row.quantity > 30]
  ];
  expect(likePrefix.length).toBeGreaterThan(0);
  for (const [predicate, matches] of cases) {
    const {execution, result} = await run(runtime, `SELECT id FROM places WHERE ${predicate}`);
    const expected = rows.filter(matches).map(row => row.id);
    expect(
      result.map(row => row.id),
      predicate
    ).toEqual(expected);
    expect(execution.materialization).toBe('resident');
  }
});

test('string projection keeps dictionary labels and aliases work', async () => {
  const {runtime} = await makeRuntime();
  const {result} = await run(runtime, "SELECT city AS town, id FROM places WHERE city LIKE 'Z%'");
  expect(result).toEqual(
    rows.filter(row => row.city?.startsWith('Z')).map(row => ({town: row.city, id: row.id}))
  );
  const star = await run(runtime, "SELECT * FROM places WHERE city = 'Beta'");
  expect(Object.keys(star.result[0])).toEqual(['id', 'city', 'amount', 'quantity']);
  expect(star.result.every(row => row.city === 'Beta')).toBe(true);
});

test('ORDER BY a string follows label order, not dictionary insertion order', async () => {
  const {runtime} = await makeRuntime();
  const {result, execution} = await run(
    runtime,
    'SELECT city, id FROM places ORDER BY city LIMIT 8'
  );
  expect(Object.keys(result[0])).toEqual(['city', 'id']);
  const expected = rows
    .filter(row => row.city !== null)
    .sort((a, b) => compareResidentSqlStrings(a.city!, b.city!) || a.id - b.id)
    .slice(0, 8);
  expect(result.map(row => row.city)).toEqual(expected.map(row => row.city));
  expect(execution.rowCount).toBe(8);
  const descending = await run(runtime, 'SELECT city, id FROM places ORDER BY city DESC LIMIT 4');
  const labels = descending.result.map(row => row.city as string);
  expect([...labels].sort((a, b) => compareResidentSqlStrings(b, a))).toEqual(labels);
});

test('GROUP BY a string key with a string predicate', async () => {
  const {runtime} = await makeRuntime();
  const {result} = await run(
    runtime,
    "SELECT city, COUNT(*) AS rows, SUM(CAST(quantity AS DOUBLE)) AS total FROM places WHERE city LIKE '%a%' GROUP BY city"
  );
  const groups = new Map<string, {rows: number; total: number}>();
  for (const row of rows) {
    if (!row.city?.includes('a')) continue;
    const group = groups.get(row.city) ?? {rows: 0, total: 0};
    group.rows++;
    group.total += row.quantity;
    groups.set(row.city, group);
  }
  expect(
    new Map(result.map(row => [row.city, {rows: Number(row.rows), total: Number(row.total)}]))
  ).toEqual(groups);
});

test('CAST among int, float and bool follows DuckDB rounding and nulls out-of-range', async () => {
  const {runtime} = await makeRuntime();
  const {result, execution} = await run(
    runtime,
    'SELECT id, CAST(quantity AS DOUBLE) AS quantity_float, CAST(amount AS INTEGER) AS amount_int, amount::BOOLEAN AS amount_flag, CAST(quantity AS UINTEGER) AS quantity_unsigned FROM places'
  );
  expect(execution.rowCount).toBe(ROW_COUNT);
  for (const row of rows) {
    const cells = result[row.id];
    expect(cells.quantity_float).toBe(row.quantity);
    const rounded =
      row.amount === null ? null : castResidentSqlValue(row.amount, 'float32', 'sint32');
    expect(cells.amount_int ?? null).toBe(rounded?.valid ? rounded.value : null);
    expect(cells.amount_flag ?? null).toBe(row.amount === null ? null : row.amount !== 0 ? 1 : 0);
    expect(cells.quantity_unsigned ?? null).toBe(row.quantity >= 0 ? row.quantity : null);
  }
  const filtered = await run(runtime, 'SELECT id FROM places WHERE CAST(amount AS INTEGER) = -4');
  expect(filtered.result.map(row => row.id)).toEqual(
    rows
      .filter(
        row =>
          row.amount !== null && Math.round(Math.abs(row.amount)) * Math.sign(row.amount) === -4
      )
      .map(row => row.id)
  );
});

test('unsupported string and CAST forms refuse with a reason instead of wrong answers', async () => {
  const {runtime} = await makeRuntime();
  expect(() => runtime.planner.query("SELECT id FROM places WHERE quantity = 'x'")).toThrow(
    /String literal/
  );
  expect(() =>
    runtime.planner.query("SELECT id FROM places WHERE soundex(city) = 'B300'")
  ).toThrow();
  expect(() => runtime.planner.query('SELECT UPPER(quantity) AS q FROM places')).toThrow();
});

const CUSTOMER_ROWS = [
  {customer: 'ann', region: 'north', score: 1.5},
  {customer: 'bob', region: 'south', score: 2.5},
  {customer: 'bob', region: 'east', score: 3.5},
  {customer: 'dan', region: 'west', score: 4.5}
];
const ORDER_ROWS = [
  {order_id: 1, customer: 'ann'},
  {order_id: 2, customer: 'bob'},
  {order_id: 3, customer: 'cat'},
  {order_id: 4, customer: 'bob'},
  {order_id: 5, customer: null as string | null}
];

test('right-side join columns gather through the GPU probe and honour duplicate policies', async () => {
  const {session, runtime} = await makeRuntime();
  session.importArrowDataset({
    id: 'orders',
    table: new arrow.Table({
      order_id: arrow.makeVector(new Uint32Array(ORDER_ROWS.map(row => row.order_id))),
      customer: arrow.vectorFromArray(ORDER_ROWS.map(row => row.customer))
    })
  });
  session.importArrowDataset({
    id: 'customers',
    table: new arrow.Table({
      customer: arrow.vectorFromArray(CUSTOMER_ROWS.map(row => row.customer)),
      region: arrow.vectorFromArray(CUSTOMER_ROWS.map(row => row.region)),
      score: arrow.vectorFromArray(
        CUSTOMER_ROWS.map(row => row.score),
        new arrow.Float32()
      )
    })
  });
  for (const policy of ['all', 'first', 'last'] as const) {
    for (const [keyword, outer] of [
      ['JOIN', false],
      ['LEFT JOIN', true]
    ] as const) {
      const {execution, result} = await run(
        runtime,
        `-- luma: duplicateRightMatches=${policy}\nSELECT o.order_id, c.region, c.score AS points FROM orders o ${keyword} customers c ON o.customer = c.customer`
      );
      const expected: Record<string, unknown>[] = [];
      for (const order of ORDER_ROWS) {
        const matches = CUSTOMER_ROWS.filter(
          row => order.customer !== null && row.customer === order.customer
        );
        const chosen =
          policy === 'first'
            ? matches.slice(0, 1)
            : policy === 'last'
              ? matches.slice(-1)
              : matches;
        for (const match of chosen)
          expected.push({order_id: order.order_id, region: match.region, points: match.score});
        if (!chosen.length && outer)
          expected.push({order_id: order.order_id, region: null, points: null});
      }
      expect(result, `${keyword} ${policy}`).toEqual(expected);
      expect(execution.rowCount).toBe(expected.length);
      await execution.discard?.();
    }
  }
  // FULL JOIN appends unmatched right rows; RIGHT JOIN keeps every right row.
  const full = await run(
    runtime,
    'SELECT o.order_id, c.customer AS buyer FROM orders o FULL JOIN customers c ON o.customer = c.customer'
  );
  expect(full.result.map(row => [row.order_id ?? null, row.buyer ?? null])).toEqual([
    [1, 'ann'],
    [2, 'bob'],
    [2, 'bob'],
    [3, null],
    [4, 'bob'],
    [4, 'bob'],
    [5, null],
    [null, 'dan']
  ]);
  await full.execution.discard?.();
  const right = await run(
    runtime,
    'SELECT c.region, o.order_id FROM orders o RIGHT JOIN customers c ON o.customer = c.customer'
  );
  expect(right.result.map(row => row.region).sort()).toEqual([
    'east',
    'east',
    'north',
    'south',
    'south',
    'west'
  ]);
  await right.execution.discard?.();
});

test('multi-batch filters gather every source batch and GROUP BY drops empty slots', async () => {
  const {runtime} = await makeRuntime();
  const filtered = await run(runtime, 'SELECT id, quantity FROM places WHERE id > 5 AND id < 14');
  expect(filtered.result).toEqual(
    rows.filter(row => row.id > 5 && row.id < 14).map(row => ({id: row.id, quantity: row.quantity}))
  );
  const grouped = await run(runtime, 'SELECT city, COUNT(*) AS rows FROM places GROUP BY city');
  const counts = new Map<string | null, number>();
  for (const row of rows)
    if (row.city !== null) counts.set(row.city, (counts.get(row.city) ?? 0) + 1);
  expect(new Map(grouped.result.map(row => [row.city as string, Number(row.rows)]))).toEqual(
    counts
  );
});
