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

type OrderRow = {
  order_id: number;
  customer: string | null;
  region: string | null;
  amount: number;
  year: number | null;
  month: number;
};
type CustomerRow = {
  customer: string | null;
  region: string | null;
  score: number;
  year: number | null;
  month: number;
};

const CUSTOMER_POOL = ['ann', 'bob', 'cat', 'dan', null];
const REGION_POOL = ['n', 's', 'e', null];
const orderRows: OrderRow[] = Array.from({length: 18}, (_, index) => ({
  order_id: index + 1,
  customer: CUSTOMER_POOL[(index * 7 + 1) % 5],
  region: REGION_POOL[(index * 3) % 4],
  amount: index * 1.5 - 5,
  year: index % 9 === 8 ? null : 2020 + (index % 3),
  month: 1 + (index % 4)
}));
const customerRows: CustomerRow[] = Array.from({length: 10}, (_, index) => ({
  customer: CUSTOMER_POOL[(index * 3) % 5],
  region: REGION_POOL[(index * 2 + 1) % 4],
  score: index * 0.5 + 1,
  year: 2020 + (index % 3),
  month: 1 + ((index * 3) % 4)
}));

async function makeRuntime() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 256 * 1024 * 1024});
  session.importArrowDataset({
    id: 'orders',
    table: new arrow.Table({
      order_id: arrow.makeVector(new Uint32Array(orderRows.map(row => row.order_id))),
      customer: arrow.vectorFromArray(orderRows.map(row => row.customer)),
      region: arrow.vectorFromArray(orderRows.map(row => row.region)),
      amount: arrow.vectorFromArray(
        orderRows.map(row => row.amount),
        new arrow.Float32()
      ),
      year: arrow.vectorFromArray(
        orderRows.map(row => row.year),
        new arrow.Int32()
      ),
      month: arrow.vectorFromArray(
        orderRows.map(row => row.month),
        new arrow.Int32()
      )
    })
  });
  session.importArrowDataset({
    id: 'customers',
    table: new arrow.Table({
      customer: arrow.vectorFromArray(customerRows.map(row => row.customer)),
      region: arrow.vectorFromArray(customerRows.map(row => row.region)),
      score: arrow.vectorFromArray(
        customerRows.map(row => row.score),
        new arrow.Float32()
      ),
      year: arrow.vectorFromArray(
        customerRows.map(row => row.year),
        new arrow.Int32()
      ),
      month: arrow.vectorFromArray(
        customerRows.map(row => row.month),
        new arrow.Int32()
      )
    })
  });
  return {session, runtime: new ResidentSqlGpuRuntime(session)};
}

async function readRows(execution: ResidentSqlExecution): Promise<Record<string, unknown>[]> {
  const resident = execution.dataset.value.resident;
  if (resident)
    await resident.ensureRows(Array.from({length: execution.rowCount}, (_, index) => index));
  return execution.dataset.value.source.toArray().map(row => row.toJSON());
}

async function run(runtime: ResidentSqlGpuRuntime, session: KeplerComputeSession, sql: string) {
  const execution = await runtime.execute(runtime.planner.query(sql));
  const result = await readRows(execution);
  // The join intermediate must never stay in the registry.
  expect(
    session.registry.getDatasetFootprints().filter(({id}) => id.startsWith('sql-gpu-join')),
    sql
  ).toEqual([]);
  return {execution, result};
}

type Pair = {order: OrderRow; customer: CustomerRow | null};
function joinOracle(
  type: 'INNER' | 'LEFT',
  matches: (order: OrderRow, customer: CustomerRow) => boolean
): Pair[] {
  const pairs: Pair[] = [];
  for (const order of orderRows) {
    const found = customerRows.filter(customer => matches(order, customer));
    for (const customer of found) pairs.push({order, customer});
    if (!found.length && type === 'LEFT') pairs.push({order, customer: null});
  }
  return pairs;
}
const sameString = (a: string | null, b: string | null) => a !== null && b !== null && a === b;
const sameNumber = (a: number | null, b: number | null) => a !== null && b !== null && a === b;
const normalize = (rows: Record<string, unknown>[]) =>
  rows.map(row => JSON.stringify(row, Object.keys(row).sort())).sort();

test('a composed join runs WHERE, GROUP BY, expressions and ORDER BY/LIMIT on the joined rows', async () => {
  const {session, runtime} = await makeRuntime();
  const pairs = joinOracle('INNER', (order, customer) =>
    sameString(order.customer, customer.customer)
  );
  expect(pairs.length).toBeGreaterThan(5);

  const grouped = await run(
    runtime,
    session,
    'SELECT c.region, COUNT(*) AS n, SUM(o.amount) AS total FROM orders o JOIN customers c ON o.customer = c.customer WHERE o.amount > 1 GROUP BY c.region'
  );
  const expectedGroups = new Map<string | null, {n: number; total: number}>();
  for (const {order, customer} of pairs) {
    if (!(order.amount > 1) || customer!.region === null) continue;
    const group = expectedGroups.get(customer!.region) ?? {n: 0, total: 0};
    group.n++;
    group.total += order.amount;
    expectedGroups.set(customer!.region, group);
  }
  expect(
    new Map(
      grouped.result.map(row => [
        row.region as string,
        {n: Number(row.n), total: Number(row.total)}
      ])
    )
  ).toEqual(expectedGroups);

  const ordered = await run(
    runtime,
    session,
    'SELECT o.order_id, c.score FROM orders o JOIN customers c ON o.customer = c.customer WHERE c.score >= 2 ORDER BY c.score DESC LIMIT 3'
  );
  const expectedOrdered = pairs
    .filter(({customer}) => customer!.score >= 2)
    .sort((a, b) => b.customer!.score - a.customer!.score)
    .slice(0, 3)
    .map(({order, customer}) => ({order_id: order.order_id, score: customer!.score}));
  expect(ordered.result.map(row => ({order_id: row.order_id, score: row.score}))).toEqual(
    expectedOrdered
  );
  expect(ordered.execution.materialization).toBe('resident');

  const expression = await run(
    runtime,
    session,
    'SELECT o.order_id, o.amount * 2 AS doubled, c.score FROM orders o LEFT JOIN customers c ON o.customer = c.customer WHERE o.order_id <= 6'
  );
  const leftPairs = joinOracle('LEFT', (order, customer) =>
    sameString(order.customer, customer.customer)
  );
  expect(normalize(expression.result)).toEqual(
    normalize(
      leftPairs
        .filter(({order}) => order.order_id <= 6)
        .map(({order, customer}) => ({
          order_id: order.order_id,
          doubled: order.amount * 2,
          score: customer ? customer.score : null
        }))
    )
  );
});

test('multi-column ON keys: exact string pairs, hashed numbers (verified) and mixed keys', async () => {
  const {session, runtime} = await makeRuntime();
  const cases: {
    name: string;
    sql: string;
    type: 'INNER' | 'LEFT';
    matches: (order: OrderRow, customer: CustomerRow) => boolean;
  }[] = [
    {
      name: 'two string keys',
      sql: 'SELECT o.order_id, c.score FROM orders o JOIN customers c ON o.customer = c.customer AND o.region = c.region',
      type: 'INNER',
      matches: (order, customer) =>
        sameString(order.customer, customer.customer) && sameString(order.region, customer.region)
    },
    {
      name: 'two string keys, LEFT',
      sql: 'SELECT o.order_id, c.score FROM orders o LEFT JOIN customers c ON o.customer = c.customer AND o.region = c.region',
      type: 'LEFT',
      matches: (order, customer) =>
        sameString(order.customer, customer.customer) && sameString(order.region, customer.region)
    },
    {
      name: 'two integer keys with a WHERE',
      sql: 'SELECT o.order_id, c.score FROM orders o JOIN customers c ON o.year = c.year AND o.month = c.month WHERE o.amount > 0',
      type: 'INNER',
      matches: (order, customer) =>
        sameNumber(order.year, customer.year) &&
        sameNumber(order.month, customer.month) &&
        order.amount > 0
    },
    {
      name: 'string and integer keys',
      sql: 'SELECT o.order_id, c.score FROM orders o JOIN customers c ON o.customer = c.customer AND o.year = c.year',
      type: 'INNER',
      matches: (order, customer) =>
        sameString(order.customer, customer.customer) && sameNumber(order.year, customer.year)
    },
    {
      name: 'three keys with USING-style names',
      sql: 'SELECT o.order_id, c.score FROM orders o JOIN customers c ON o.customer = c.customer AND o.year = c.year AND o.month = c.month',
      type: 'INNER',
      matches: (order, customer) =>
        sameString(order.customer, customer.customer) &&
        sameNumber(order.year, customer.year) &&
        sameNumber(order.month, customer.month)
    }
  ];
  for (const testCase of cases) {
    const {result} = await run(runtime, session, testCase.sql);
    const expected = joinOracle(testCase.type, testCase.matches).map(({order, customer}) => ({
      order_id: order.order_id,
      score: customer ? customer.score : null
    }));
    expect(expected.length, testCase.name).toBeGreaterThan(0);
    expect(normalize(result), testCase.name).toEqual(normalize(expected));
  }
});

test('numeric multi-column keys support LEFT, RIGHT, FULL, SEMI and ANTI through exact ranks', async () => {
  const {session, runtime} = await makeRuntime();
  const matchesPair = (order: OrderRow, customer: CustomerRow) =>
    sameNumber(order.year, customer.year) && sameNumber(order.month, customer.month);
  const columns = 'SELECT o.order_id, c.score FROM orders o';
  const on = 'ON o.year = c.year AND o.month = c.month';
  const left = await run(runtime, session, `${columns} LEFT JOIN customers c ${on}`);
  const leftExpected = joinOracle('LEFT', matchesPair).map(({order, customer}) => ({
    order_id: order.order_id,
    score: customer ? customer.score : null
  }));
  expect(normalize(left.result)).toEqual(normalize(leftExpected));
  expect(leftExpected.some(row => row.score === null)).toBe(true);
  const rightExpected = customerRows.flatMap(customer => {
    const found = orderRows.filter(order => matchesPair(order, customer));
    return found.length
      ? found.map(order => ({order_id: order.order_id, score: customer.score}))
      : [{order_id: null, score: customer.score}];
  });
  const right = await run(runtime, session, `${columns} RIGHT JOIN customers c ${on}`);
  expect(normalize(right.result)).toEqual(normalize(rightExpected));
  const matchedCustomers = new Set(
    customerRows.filter(customer => orderRows.some(order => matchesPair(order, customer)))
  );
  const fullExpected = [
    ...leftExpected,
    ...customerRows
      .filter(customer => !matchedCustomers.has(customer))
      .map(customer => ({order_id: null, score: customer.score}))
  ];
  const full = await run(runtime, session, `${columns} FULL JOIN customers c ${on}`);
  expect(normalize(full.result)).toEqual(normalize(fullExpected));
  const hasMatch = (order: OrderRow) => customerRows.some(customer => matchesPair(order, customer));
  const semi = await run(
    runtime,
    session,
    'SELECT order_id FROM orders o SEMI JOIN customers c ON o.year = c.year AND o.month = c.month'
  );
  expect(normalize(semi.result)).toEqual(
    normalize(orderRows.filter(hasMatch).map(order => ({order_id: order.order_id})))
  );
  const anti = await run(
    runtime,
    session,
    'SELECT order_id FROM orders o ANTI JOIN customers c ON o.year = c.year AND o.month = c.month'
  );
  expect(normalize(anti.result)).toEqual(
    normalize(
      orderRows.filter(order => !hasMatch(order)).map(order => ({order_id: order.order_id}))
    )
  );
  // Mixed string and numeric keys are exact too, with first-match duplicate policy.
  const mixed = await run(
    runtime,
    session,
    'SELECT o.order_id, c.score FROM orders o LEFT JOIN customers c ON o.customer = c.customer AND o.year = c.year'
  );
  expect(normalize(mixed.result)).toEqual(
    normalize(
      joinOracle(
        'LEFT',
        (order, customer) =>
          sameString(order.customer, customer.customer) && sameNumber(order.year, customer.year)
      ).map(({order, customer}) => ({
        order_id: order.order_id,
        score: customer ? customer.score : null
      }))
    )
  );
});

test('an ANTI join can select right columns, which are NULL on every kept row', async () => {
  const {session, runtime} = await makeRuntime();
  const result = await run(
    runtime,
    session,
    'SELECT o.order_id, c.score FROM orders o ANTI JOIN customers c ON o.customer = c.customer'
  );
  const kept = orderRows.filter(
    order => !customerRows.some(customer => sameString(order.customer, customer.customer))
  );
  expect(result.result.map(row => row.order_id)).toEqual(kept.map(order => order.order_id));
  expect(result.result.every(row => row.score === null || row.score === undefined)).toBe(true);
});

test('SEMI and ANTI joins keep left rows only, with DuckDB NULL semantics', async () => {
  const {session, runtime} = await makeRuntime();
  const hasMatch = (order: OrderRow) =>
    customerRows.some(customer => sameString(order.customer, customer.customer));
  const semi = await run(
    runtime,
    session,
    'SELECT order_id, amount FROM orders SEMI JOIN customers ON orders.customer = customers.customer WHERE amount > 0'
  );
  expect(normalize(semi.result)).toEqual(
    normalize(
      orderRows
        .filter(order => hasMatch(order) && order.amount > 0)
        .map(order => ({order_id: order.order_id, amount: order.amount}))
    )
  );
  const anti = await run(
    runtime,
    session,
    'SELECT order_id, customer FROM orders ANTI JOIN customers ON orders.customer = customers.customer'
  );
  expect(normalize(anti.result)).toEqual(
    normalize(
      orderRows
        .filter(order => !hasMatch(order))
        .map(order => ({order_id: order.order_id, customer: order.customer}))
    )
  );
  // The NULL-customer orders are kept by ANTI.
  expect(anti.result.some(row => row.customer === null || row.customer === undefined)).toBe(true);
  const hasPairMatch = (order: OrderRow) =>
    customerRows.some(
      customer =>
        sameString(order.customer, customer.customer) && sameString(order.region, customer.region)
    );
  const antiPair = await run(
    runtime,
    session,
    'SELECT order_id FROM orders ANTI JOIN customers ON orders.customer = customers.customer AND orders.region = customers.region ORDER BY order_id LIMIT 5'
  );
  expect(antiPair.result.map(row => row.order_id)).toEqual(
    orderRows
      .filter(order => !hasPairMatch(order))
      .map(order => order.order_id)
      .slice(0, 5)
  );
  const semiPair = await run(
    runtime,
    session,
    'SELECT COUNT(*) AS n FROM orders SEMI JOIN customers ON orders.customer = customers.customer AND orders.region = customers.region'
  );
  expect(Number(semiPair.result[0].n)).toBe(orderRows.filter(hasPairMatch).length);
});

test('unsupported composed forms refuse at plan time with a reason', async () => {
  const {runtime} = await makeRuntime();
  expect(() =>
    runtime.planner.query(
      'SELECT c.score FROM orders o SEMI JOIN customers c ON o.customer = c.customer'
    )
  ).toThrow(/only left columns/);
  expect(() =>
    runtime.planner.query(
      'SELECT o.order_id FROM orders o JOIN customers c ON o.customer = c.customer JOIN orders p ON p.order_id = o.order_id'
    )
  ).toThrow();
});
