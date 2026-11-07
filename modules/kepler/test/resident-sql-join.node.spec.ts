// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  DEFAULT_RESIDENT_SQL_JOIN_POLICY,
  makeResidentSqlJoinSource,
  parseResidentSqlJoin,
  planResidentSqlJoin,
  readResidentSqlJoinPolicyDirective,
  type ResidentSqlJoinSource,
  type ResidentSqlJoinStatement
} from '../../../examples/deck/kepler-compute/src/resident-sql-join';
import {
  RESIDENT_JOIN_NO_ROW,
  makeResidentJoinDictionaryKeyTables,
  planResidentJoinPairs
} from '../../../examples/deck/kepler-compute/src/resident-join-plan';
import {planResidentJoinPairsOnGpu} from '../../../examples/deck/kepler-compute/src/resident-join-gpu-reference';

function makeSnapshot(
  id: string,
  names: string[],
  formats: Record<string, string>,
  dictionaries: Record<string, string[]>,
  rowCount: number
) {
  return {
    id,
    revision: 1,
    rowCount,
    value: {
      source: {schema: {fields: names.map(name => ({name}))}},
      columns: Object.fromEntries(
        Object.entries(formats).map(([name, format]) => [name, {format}])
      ),
      prepared: {
        dictionaries: Object.fromEntries(
          Object.entries(dictionaries).map(([name, values]) => [name, {values, ordered: false}])
        )
      },
      analytics: {nullCounts: {}}
    }
  };
}

const ORDER_CUSTOMERS = ['ann', 'bob', 'cat'];
const CUSTOMER_KEYS = ['ann', 'bob', 'bob', 'dan'];
const orders = makeResidentSqlJoinSource(
  makeSnapshot(
    'orders',
    ['order_id', 'customer', 'amount'],
    {order_id: 'uint32', customer: 'uint32', amount: 'float32'},
    {customer: ORDER_CUSTOMERS},
    5
  )
);
const customers = makeResidentSqlJoinSource(
  makeSnapshot(
    'customers',
    ['customer', 'region', 'score', 'geometry'],
    {customer: 'uint32', region: 'uint32', score: 'float32'},
    {customer: CUSTOMER_KEYS, region: ['north', 'south']},
    4
  )
);
const sources: Record<string, ResidentSqlJoinSource> = {orders, customers};
const findSource = (name: string) => sources[name];

function parse(sql: string): ResidentSqlJoinStatement {
  const parsed = parseResidentSqlJoin(sql);
  if (!parsed || 'reason' in parsed) throw new Error(`not parsed: ${JSON.stringify(parsed)}`);
  return parsed;
}

test('join statements parse; others report why the executor cannot run them', () => {
  expect(parseResidentSqlJoin('SELECT a FROM t')).toBeNull();
  const statement = parse(
    'SELECT o.order_id, c.region AS area FROM orders o LEFT OUTER JOIN customers AS c ON o.customer = c.customer;'
  );
  expect(statement.type).toBe('LEFT');
  expect(statement.left).toEqual({table: 'orders', alias: 'o'});
  expect(statement.right).toEqual({table: 'customers', alias: 'c'});
  expect(statement.items).toHaveLength(2);
  expect(parse('SELECT * FROM orders JOIN customers USING (customer)').using).toBe(true);
  expect(
    parse('SELECT o.*, c.region FROM orders o RIGHT JOIN customers c ON c.customer = o.customer')
      .type
  ).toBe('RIGHT');
  const reasonOf = (sql: string) => (parseResidentSqlJoin(sql) as {reason: string}).reason;
  expect(
    reasonOf(
      'SELECT * FROM orders a JOIN customers b ON a.customer = b.customer WHERE a.order_id > 1'
    )
  ).toMatch(/WHERE after a join/);
  expect(
    reasonOf('SELECT count(*) FROM orders a JOIN customers b ON a.customer = b.customer')
  ).toMatch(/columns or \*/);
  expect(
    reasonOf('SELECT * FROM orders a SEMI JOIN customers b ON a.customer = b.customer')
  ).toMatch(/SEMI JOIN has no right columns/);
  expect(reasonOf('SELECT * FROM orders a JOIN customers b ON a.customer > b.customer')).toMatch(
    /single column equality/
  );
});

test('directive comments set the duplicate and naming policy', () => {
  expect(
    readResidentSqlJoinPolicyDirective(
      '-- luma: duplicateRightMatches=first, rightColumnSuffix=_r, dropRightKey\nSELECT 1'
    )
  ).toEqual({duplicateRightMatches: 'first', rightColumnSuffix: '_r', dropRightKey: true});
  expect(readResidentSqlJoinPolicyDirective('SELECT 1')).toEqual({});
  expect(() => readResidentSqlJoinPolicyDirective('-- luma: duplicateRightMatches=some')).toThrow(
    /all, first or last/
  );
  expect(() => readResidentSqlJoinPolicyDirective('-- luma: nope=1')).toThrow(/Unknown/);
});

test('right columns are projected in select order with SQL names and colliding suffixes', () => {
  const planned = planResidentSqlJoin({
    statement: parse(
      'SELECT c.region, o.order_id, c.score AS points, o.customer, c.customer FROM orders o JOIN customers c ON o.customer = c.customer'
    ),
    findSource,
    policy: DEFAULT_RESIDENT_SQL_JOIN_POLICY,
    resultId: 'sql-gpu-result-1'
  });
  expect(
    planned.plan.columns.map(column => [column.side, column.sourceColumn, column.outputColumn])
  ).toEqual([
    ['right', 'region', 'region'],
    ['left', 'order_id', 'order_id'],
    ['right', 'score', 'points'],
    ['left', 'customer', 'customer'],
    ['right', 'customer', 'customer_right']
  ]);
  expect(planned.plan.provenance.columns.map(column => column.name)).toEqual(
    planned.plan.columns.map(column => column.outputColumn)
  );
  expect(planned.plan.provenance.fields.map(field => field.name)).toEqual(
    planned.plan.columns.map(column => column.outputColumn)
  );
  expect(planned.plan.joinType).toBe('INNER');
  expect(planned.plan.keyKind).toBe('string');
  expect(planned.plan.columns[0].dictionary).toBe(true);
  expect(planned.plan.columns[2].format).toBe('float32');
});

test('star, USING and dropRightKey drop the repeated right key; RIGHT JOIN swaps sides', () => {
  const using = planResidentSqlJoin({
    statement: parse('SELECT * FROM orders JOIN customers USING (customer)'),
    findSource: name =>
      name === 'customers'
        ? {...customers, columnNames: ['customer', 'region', 'score']}
        : sources[name],
    policy: DEFAULT_RESIDENT_SQL_JOIN_POLICY,
    resultId: 'r'
  });
  expect(using.outputs.map(output => output.name)).toEqual([
    'order_id',
    'customer',
    'amount',
    'region',
    'score'
  ]);
  const right = planResidentSqlJoin({
    statement: parse(
      'SELECT o.order_id, c.region FROM orders o RIGHT JOIN customers c ON o.customer = c.customer'
    ),
    findSource,
    policy: DEFAULT_RESIDENT_SQL_JOIN_POLICY,
    resultId: 'r'
  });
  expect(right.swapped).toBe(true);
  expect(right.plan.joinType).toBe('LEFT');
  expect(right.plan.left.datasetId).toBe('customers');
  expect(right.plan.columns.map(column => [column.side, column.sourceColumn])).toEqual([
    ['right', 'order_id'],
    ['left', 'region']
  ]);
});

test('unsupported joins refuse with a reason', () => {
  const plan = (sql: string, source = findSource) =>
    planResidentSqlJoin({
      statement: parse(sql),
      findSource: source,
      policy: DEFAULT_RESIDENT_SQL_JOIN_POLICY,
      resultId: 'r'
    });
  // SQL self-joins read one dataset on both sides and are planned like any other join.
  expect(
    plan('SELECT o.order_id FROM orders o JOIN orders p ON o.customer = p.customer').plan.left
      .datasetId
  ).toBe('orders');
  expect(() =>
    plan('SELECT o.order_id FROM orders o JOIN customers c ON o.order_id = c.customer')
  ).toThrow(/both be string dictionaries or both be numbers/);
  // geometry has no resident storage, so SELECT * cannot carry it.
  expect(() => plan('SELECT * FROM orders o JOIN customers c ON o.customer = c.customer')).toThrow(
    /cannot carry right column geometry/
  );
  expect(() =>
    plan('SELECT o.nope FROM orders o JOIN customers c ON o.customer = c.customer')
  ).toThrow(/does not exist/);
  expect(() =>
    plan('SELECT customer FROM orders o JOIN customers c ON o.customer = c.customer')
  ).toThrow(/ambiguous/);
  expect(() =>
    plan('SELECT o.customer, o.customer FROM orders o JOIN customers c ON o.customer = c.customer')
  ).toThrow(/selected twice/);
  expect(() =>
    plan('SELECT x.order_id FROM orders o JOIN customers c ON o.customer = c.customer')
  ).toThrow(/Unknown table qualifier/);
  expect(() =>
    plan('SELECT o.order_id FROM orders o JOIN missing c ON o.customer = c.customer')
  ).toThrow(/not registered/);
});

/** Independent SQL join semantics by nested loops; right matches in right source order. */
function bruteForceJoin(
  leftKeys: (string | null)[],
  rightKeys: (string | null)[],
  rightValues: number[],
  type: 'INNER' | 'LEFT',
  policy: 'all' | 'first' | 'last'
): (number | null)[] {
  const result: (number | null)[] = [];
  for (const leftKey of leftKeys) {
    const matches = rightKeys
      .map((key, row) => (leftKey !== null && leftKey !== '' && key === leftKey ? row : -1))
      .filter(row => row >= 0);
    const chosen =
      policy === 'first' ? matches.slice(0, 1) : policy === 'last' ? matches.slice(-1) : matches;
    if (chosen.length) for (const row of chosen) result.push(rightValues[row]);
    else if (type === 'LEFT') result.push(null);
  }
  return result;
}

test('duplicateRightMatches policies: oracle, GPU mirror and nested loops agree on right columns', () => {
  const leftCodes = Uint32Array.of(0, 1, 2, 1, 0);
  const rightCodes = Uint32Array.of(0, 1, 2, 3);
  const leftKeys = Array.from(leftCodes, code => ORDER_CUSTOMERS[code]);
  const rightKeys = Array.from(rightCodes, code => CUSTOMER_KEYS[code]);
  const rightValues = [10, 20, 30, 40];
  const tables = makeResidentJoinDictionaryKeyTables(ORDER_CUSTOMERS, CUSTOMER_KEYS);
  for (const policy of ['all', 'first', 'last'] as const) {
    for (const [sqlType, joinType] of [
      ['JOIN', 'INNER'],
      ['LEFT JOIN', 'LEFT']
    ] as const) {
      const planned = planResidentSqlJoin({
        statement: parse(
          `SELECT o.order_id, c.score FROM orders o ${sqlType} customers c ON o.customer = c.customer`
        ),
        findSource,
        policy: {...DEFAULT_RESIDENT_SQL_JOIN_POLICY, duplicateRightMatches: policy},
        resultId: 'r'
      });
      expect(planned.plan.duplicateRightMatches).toBe(policy);
      expect(planned.plan.provenance.duplicateRightMatches).toBe(
        policy === 'all' ? undefined : policy
      );
      const oracle = planResidentJoinPairs({
        joinType,
        leftCodes,
        leftLabels: ORDER_CUSTOMERS,
        rightCodes,
        rightLabels: CUSTOMER_KEYS,
        duplicateRightMatches: policy
      });
      const mirror = planResidentJoinPairsOnGpu({
        joinType,
        leftKey: Uint32Array.from(leftCodes, code => tables.left[code]),
        leftOk: Uint32Array.from(leftCodes, code => (tables.left[code] === 0xffffffff ? 0 : 1)),
        rightKey: Uint32Array.from(rightCodes, code => tables.right[code]),
        rightOk: Uint32Array.from(rightCodes, code => (tables.right[code] === 0xffffffff ? 0 : 1)),
        duplicateRightMatches: policy
      });
      expect(Array.from(mirror.leftRows)).toEqual(Array.from(oracle.leftRows));
      expect(Array.from(mirror.rightRows)).toEqual(Array.from(oracle.rightRows));
      const gathered = Array.from(oracle.rightRows, row =>
        row === RESIDENT_JOIN_NO_ROW ? null : rightValues[row]
      );
      expect(gathered, `${joinType} ${policy}`).toEqual(
        bruteForceJoin(leftKeys, rightKeys, rightValues, joinType, policy)
      );
    }
  }
});
