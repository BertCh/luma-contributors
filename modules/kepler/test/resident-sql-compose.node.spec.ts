// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  computeResidentJoinCompositeKeys,
  finishResidentJoinCompositeHash,
  makeResidentJoinCompositeKeyPlan,
  makeResidentJoinRankedCompositeKeyPlan,
  mixResidentJoinCompositeHash,
  type ResidentJoinCompositeKeyPlan
} from '../../../examples/deck/kepler-compute/src/resident-join-composite-key';
import {planResidentJoinPairsOnGpu} from '../../../examples/deck/kepler-compute/src/resident-join-gpu-reference';
import {RESIDENT_JOIN_NO_ROW} from '../../../examples/deck/kepler-compute/src/resident-join-plan';
import {
  parseResidentSqlCompose,
  planResidentSqlCompose,
  type ResidentSqlComposeStatement
} from '../../../examples/deck/kepler-compute/src/resident-sql-compose';
import {
  DEFAULT_RESIDENT_SQL_JOIN_POLICY,
  makeResidentSqlJoinSource,
  type ResidentSqlJoinSource
} from '../../../examples/deck/kepler-compute/src/resident-sql-join';

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

const orders = makeResidentSqlJoinSource(
  makeSnapshot(
    'orders',
    ['order_id', 'customer', 'amount', 'region'],
    {order_id: 'uint32', customer: 'uint32', amount: 'float32', region: 'uint32'},
    {customer: ['ann', 'bob', 'cat'], region: ['n', 's']},
    5
  )
);
const customers = makeResidentSqlJoinSource(
  makeSnapshot(
    'customers',
    ['customer', 'region', 'score', 'tier'],
    {customer: 'uint32', region: 'uint32', score: 'float32', tier: 'sint32'},
    {customer: ['ann', 'bob', 'bob', 'dan'], region: ['s', 'n', 'w']},
    4
  )
);
const ledgerA = makeResidentSqlJoinSource(
  makeSnapshot(
    'ledger_a',
    ['year', 'month', 'amount'],
    {year: 'sint32', month: 'sint32', amount: 'float32'},
    {},
    6
  )
);
const ledgerB = makeResidentSqlJoinSource(
  makeSnapshot(
    'ledger_b',
    ['year', 'month', 'budget'],
    {year: 'sint32', month: 'sint32', budget: 'float32'},
    {},
    6
  )
);
const sources: Record<string, ResidentSqlJoinSource> = {
  orders,
  customers,
  ledger_a: ledgerA,
  ledger_b: ledgerB
};

function parse(sql: string): ResidentSqlComposeStatement {
  const parsed = parseResidentSqlCompose(sql);
  if (!parsed || 'reason' in parsed) throw new Error(`not parsed: ${JSON.stringify(parsed)}`);
  return parsed;
}
function plan(sql: string, policy = DEFAULT_RESIDENT_SQL_JOIN_POLICY) {
  return planResidentSqlCompose({
    statement: parse(sql),
    findSource: name => sources[name],
    policy,
    resultId: 'joined'
  });
}

test('composed join heads parse; unsupported shapes report a reason', () => {
  expect(parseResidentSqlCompose('SELECT a FROM t')).toBeNull();
  const statement = parse(
    'SELECT count(*) FROM orders o LEFT JOIN customers c ON o.customer = c.customer AND o.region = c.region WHERE o.amount > 1'
  );
  expect(statement.type).toBe('LEFT');
  expect(statement.keys).toHaveLength(2);
  expect(parse('SELECT * FROM orders SEMI JOIN customers USING (customer, region)').using).toBe(
    true
  );
  expect(
    parse('SELECT order_id FROM orders ANTI JOIN customers ON orders.customer = customers.customer')
      .type
  ).toBe('ANTI');
  expect(parse('SELECT 1 FROM orders LEFT SEMI JOIN customers USING (customer)').type).toBe('SEMI');
  const reasonOf = (sql: string) => (parseResidentSqlCompose(sql) as {reason: string}).reason;
  expect(reasonOf('SELECT * FROM orders a JOIN customers b ON a.customer > b.customer')).toMatch(
    /column = column/
  );
  expect(reasonOf('SELECT * FROM orders a CROSS JOIN customers b')).toMatch(/CROSS/);
  expect(
    reasonOf(
      'SELECT * FROM orders a JOIN customers b ON a.customer = b.customer JOIN ledger_a c ON c.year = a.order_id'
    )
  ).toMatch(/more than two tables|JOIN after/);
  expect(
    reasonOf(
      'SELECT * FROM orders a JOIN customers b ON a.customer = b.customer AND a.region = b.region AND a.order_id = b.tier AND a.amount = b.score AND a.order_id = b.tier'
    )
  ).toMatch(/at most 4/);
  expect(
    reasonOf(
      'SELECT * FROM orders a JOIN customers b ON a.customer = b.customer WHERE a.order_id IN (SELECT 1)'
    )
  ).toMatch(/Subqueries/);
});

test('trailing clauses are rewritten over the intermediate table', () => {
  const grouped = plan(
    'SELECT c.region, COUNT(*) AS n, SUM(o.amount) AS total FROM orders o JOIN customers c ON o.customer = c.customer WHERE o.amount > 1 GROUP BY c.region ORDER BY n DESC LIMIT 3'
  );
  expect(grouped.outerSql).toBe(
    'SELECT region, COUNT(*) AS n, SUM(amount) AS total FROM luma_sql_joined WHERE amount > 1 GROUP BY region ORDER BY n DESC LIMIT 3'
  );
  expect(grouped.joinPlan.outputs.map(output => `${output.side}.${output.name}`)).toEqual([
    'left.amount',
    'right.region'
  ]);
  const collided = plan(
    'SELECT o.region, c.region FROM orders o JOIN customers c ON o.customer = c.customer WHERE c.score > 1'
  );
  expect(collided.outerSql).toBe(
    'SELECT region, region_right FROM luma_sql_joined WHERE score > 1'
  );
  const renamed = plan(
    "SELECT c.region FROM orders o JOIN customers c ON o.customer = c.customer WHERE o.region = 'n'"
  );
  expect(renamed.outerSql).toBe(
    "SELECT region_right AS region FROM luma_sql_joined WHERE region = 'n'"
  );
  const suffixed = plan(
    'SELECT o.region, c.region FROM orders o JOIN customers c ON o.customer = c.customer',
    {...DEFAULT_RESIDENT_SQL_JOIN_POLICY, rightColumnSuffix: '_c'}
  );
  expect(suffixed.outerSql).toContain('region_c');
  const star = plan(
    'SELECT * FROM orders o JOIN customers c USING (customer) ORDER BY score DESC LIMIT 2'
  );
  expect(star.outerSql).toBe(
    'SELECT order_id, customer, amount, region, region_right, score, tier FROM luma_sql_joined ORDER BY score DESC LIMIT 2'
  );
  const swapped = plan(
    'SELECT o.order_id, c.score FROM orders o RIGHT JOIN customers c ON o.customer = c.customer WHERE c.score >= 2'
  );
  expect(swapped.joinPlan.swapped).toBe(true);
  expect(swapped.joinPlan.plan.joinType).toBe('LEFT');
  expect(swapped.joinPlan.plan.left.datasetId).toBe('customers');
});

test('ambiguous or unknown names and right columns of SEMI/ANTI are refused', () => {
  expect(() =>
    plan(
      'SELECT region FROM orders o JOIN customers c ON o.customer = c.customer WHERE o.amount > 1'
    )
  ).toThrow(/ambiguous/);
  expect(() =>
    plan('SELECT nope FROM orders o JOIN customers c ON o.customer = c.customer WHERE o.amount > 1')
  ).toThrow(/does not exist/);
  expect(() =>
    plan('SELECT c.score FROM orders o SEMI JOIN customers c ON o.customer = c.customer')
  ).toThrow(/only left columns/);
  // SEMI never exposes right columns; an ANTI join can select them (they are NULL on every kept row).
  expect(() =>
    plan('SELECT score FROM orders o SEMI JOIN customers c ON o.customer = c.customer')
  ).toThrow(/does not exist/);
  const antiRight = plan(
    'SELECT o.order_id, c.score FROM orders o ANTI JOIN customers c ON o.customer = c.customer'
  );
  expect(antiRight.joinPlan.plan.columns.some(column => column.side === 'right')).toBe(true);
  expect(antiRight.outerSql).toContain('IS NULL');
});

test('SEMI and ANTI reuse the executor with first-match and a right-key probe', () => {
  const semi = plan(
    'SELECT order_id FROM orders SEMI JOIN customers ON orders.customer = customers.customer WHERE order_id > 1'
  );
  expect(semi.outerSql).toBe('SELECT order_id FROM luma_sql_joined WHERE order_id > 1');
  expect(semi.joinPlan.plan.joinType).toBe('INNER');
  expect(semi.joinPlan.plan.duplicateRightMatches).toBe('first');
  expect(semi.joinPlan.plan.columns.every(column => column.side === 'left')).toBe(true);
  const anti = plan(
    'SELECT * FROM orders ANTI JOIN customers ON orders.customer = customers.customer'
  );
  expect(anti.joinPlan.plan.joinType).toBe('LEFT');
  expect(anti.joinPlan.plan.duplicateRightMatches).toBe('first');
  expect(anti.outerSql).toBe(
    'SELECT order_id, customer, amount, region FROM luma_sql_joined WHERE customer_right IS NULL'
  );
  const antiWhere = plan(
    'SELECT order_id FROM orders ANTI JOIN customers ON orders.customer = customers.customer WHERE amount > 1 ORDER BY order_id LIMIT 2'
  );
  expect(antiWhere.outerSql).toBe(
    'SELECT order_id FROM luma_sql_joined WHERE customer IS NULL AND ( amount > 1) ORDER BY order_id LIMIT 2'
  );
});

test('multi-column keys: strings are exact, numbers are hashed and verified', () => {
  const strings = plan(
    'SELECT o.order_id FROM orders o JOIN customers c ON o.customer = c.customer AND o.region = c.region'
  );
  expect(strings.compositeKey?.exact).toBe(true);
  expect(strings.compositeKey?.radixProduct).toBe(4 * 3);
  expect(strings.outerSql).toBe('SELECT order_id FROM luma_sql_joined');
  const numbers = plan(
    'SELECT a.amount, b.budget FROM ledger_a a JOIN ledger_b b ON a.year = b.year AND a.month = b.month WHERE a.amount > 0'
  );
  expect(numbers.compositeKey?.exact).toBe(false);
  expect(numbers.outerSql).toBe(
    'SELECT amount, budget FROM luma_sql_joined WHERE __luma_sql_join_match = 1 AND ( amount > 0)'
  );
  const using = plan('SELECT count(*) FROM ledger_a JOIN ledger_b USING (year, month)');
  expect(using.outerSql).toBe(
    'SELECT count(*) FROM luma_sql_joined WHERE __luma_sql_join_match = 1'
  );
  // Other join types and duplicate policies plan against exact ranks resolved at execution: their
  // outer statement never carries the hash verification residual.
  for (const joinType of ['LEFT', 'RIGHT', 'FULL', 'SEMI', 'ANTI']) {
    const ranked = plan(
      `SELECT count(*) FROM ledger_a a ${joinType} JOIN ledger_b b ON a.year = b.year AND a.month = b.month`
    );
    expect(ranked.outerSql, joinType).toBe(ranked.exactOuterSql);
    expect(ranked.outerSql).not.toContain('__luma_sql_join_match');
  }
  expect(
    plan('SELECT count(*) FROM ledger_a a JOIN ledger_b b ON a.year = b.year AND a.month = b.month')
      .exactOuterSql
  ).toBe('SELECT count(*) FROM luma_sql_joined');
  expect(
    plan(
      'SELECT o.order_id FROM orders o LEFT JOIN customers c ON o.customer = c.customer AND o.region = c.region'
    ).joinPlan.plan.joinType
  ).toBe('LEFT');
});

test('a join side with no rows is refused by the planner', () => {
  const empty = makeResidentSqlJoinSource(
    makeSnapshot('empty', ['customer'], {customer: 'uint32'}, {customer: []}, 0)
  );
  expect(() =>
    planResidentSqlCompose({
      statement: parse(
        'SELECT o.order_id FROM orders o JOIN empty e ON o.customer = e.customer WHERE o.order_id > 0'
      ),
      findSource: name => ({...sources, empty})[name],
      policy: DEFAULT_RESIDENT_SQL_JOIN_POLICY,
      resultId: 'joined'
    })
  ).toThrow(/at least one row/);
});

// ---- CPU mirror versus a nested-loop oracle -----------------------------------------------

type KeyColumn = {
  kind: 'string' | 'numeric';
  /** String labels or numbers per row; null is SQL NULL. */
  values: (string | number | null)[];
  dictionary?: string[];
};

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function makeStringColumn(random: () => number, dictionary: string[], count: number): KeyColumn {
  return {
    kind: 'string',
    dictionary,
    values: Array.from({length: count}, () => {
      const pick = random();
      return pick < 0.12 ? null : dictionary[Math.floor(random() * dictionary.length)];
    })
  };
}
function makeNumberColumn(random: () => number, count: number, range: number): KeyColumn {
  return {
    kind: 'numeric',
    values: Array.from({length: count}, () =>
      random() < 0.12 ? null : Math.floor(random() * range) - 1
    )
  };
}
function toStorage(column: KeyColumn, format: 'sint32' | 'float32') {
  const words = new Uint32Array(column.values.length);
  const validity = new Uint32Array(column.values.length);
  const floats = new Float32Array(words.buffer);
  const ints = new Int32Array(words.buffer);
  column.values.forEach((value, row) => {
    if (value === null) return;
    validity[row] = 1;
    if (column.kind === 'string') words[row] = column.dictionary!.indexOf(value as string);
    else if (format === 'float32') floats[row] = value as number;
    else ints[row] = value as number;
  });
  return {words, validity};
}

function makeSource(
  id: string,
  columns: KeyColumn[],
  format: 'sint32' | 'float32',
  rowCount: number
) {
  const names = columns.map((_, index) => `k${index}`);
  return makeResidentSqlJoinSource(
    makeSnapshot(
      id,
      names,
      Object.fromEntries(
        columns.map((column, index) => [names[index], column.kind === 'string' ? 'uint32' : format])
      ),
      Object.fromEntries(
        columns.flatMap((column, index) =>
          column.kind === 'string' ? [[names[index], column.dictionary!]] : []
        )
      ),
      rowCount
    )
  );
}

function nestedLoopPairs(
  joinType: 'INNER' | 'LEFT' | 'FULL',
  left: KeyColumn[],
  right: KeyColumn[]
): string[] {
  const leftCount = left[0].values.length;
  const rightCount = right[0].values.length;
  const matches = (leftRow: number, rightRow: number) =>
    left.every((column, index) => {
      const a = column.values[leftRow];
      const b = right[index].values[rightRow];
      return a !== null && b !== null && a !== '' && b !== '' && a === b;
    });
  const pairs: string[] = [];
  const rightMatched = new Set<number>();
  for (let leftRow = 0; leftRow < leftCount; leftRow++) {
    let any = false;
    for (let rightRow = 0; rightRow < rightCount; rightRow++)
      if (matches(leftRow, rightRow)) {
        pairs.push(`${leftRow}:${rightRow}`);
        rightMatched.add(rightRow);
        any = true;
      }
    if (!any && joinType !== 'INNER') pairs.push(`${leftRow}:-`);
  }
  if (joinType === 'FULL')
    for (let rightRow = 0; rightRow < rightCount; rightRow++)
      if (!rightMatched.has(rightRow)) pairs.push(`-:${rightRow}`);
  return pairs.sort();
}

function mirrorPairs(props: {
  plan: ResidentJoinCompositeKeyPlan;
  joinType: 'INNER' | 'LEFT' | 'FULL';
  left: KeyColumn[];
  right: KeyColumn[];
  format: 'sint32' | 'float32';
  duplicateRightMatches?: 'all' | 'first' | 'last';
}): string[] {
  const {plan: keyPlan, left, right, format} = props;
  const leftStorage = left.map(column => toStorage(column, format));
  const rightStorage = right.map(column => toStorage(column, format));
  const leftKeys = computeResidentJoinCompositeKeys(keyPlan, 'left', leftStorage);
  const rightKeys = computeResidentJoinCompositeKeys(keyPlan, 'right', rightStorage);
  expect(leftKeys.status | rightKeys.status).toBe(0);
  const result = planResidentJoinPairsOnGpu({
    joinType: props.joinType,
    leftKey: leftKeys.key,
    leftOk: leftKeys.ok,
    rightKey: rightKeys.key,
    rightOk: rightKeys.ok,
    duplicateRightMatches: props.duplicateRightMatches
  });
  const pairs: string[] = [];
  for (let index = 0; index < result.rowCount; index++) {
    const leftRow = result.leftRows[index];
    const rightRow = result.rightRows[index];
    // The composed WHERE: numeric components must be equal in the joined row (INNER only).
    if (
      leftRow !== RESIDENT_JOIN_NO_ROW &&
      rightRow !== RESIDENT_JOIN_NO_ROW &&
      !keyPlan.exact &&
      !keyPlan.components.every(
        (component, componentIndex) =>
          component.kind === 'string' ||
          leftStorage[componentIndex].words[leftRow] ===
            rightStorage[componentIndex].words[rightRow]
      )
    )
      continue;
    pairs.push(
      `${leftRow === RESIDENT_JOIN_NO_ROW ? '-' : leftRow}:${rightRow === RESIDENT_JOIN_NO_ROW ? '-' : rightRow}`
    );
  }
  return pairs.sort();
}

function makeKeyPlan(left: KeyColumn[], right: KeyColumn[], format: 'sint32' | 'float32') {
  const leftSource = makeSource('left', left, format, left[0].values.length);
  const rightSource = makeSource('right', right, format, right[0].values.length);
  return makeResidentJoinCompositeKeyPlan({
    pairs: left.map((_, index) => ({leftColumn: `k${index}`, rightColumn: `k${index}`})),
    left: leftSource.metadata,
    right: rightSource.metadata
  });
}

test('hash helpers are deterministic 32-bit mixes', () => {
  expect(mixResidentJoinCompositeHash(0x811c9dc5, 7)).toBe(
    mixResidentJoinCompositeHash(0x811c9dc5, 7)
  );
  expect(mixResidentJoinCompositeHash(0x811c9dc5, 7)).not.toBe(
    mixResidentJoinCompositeHash(0x811c9dc5, 8)
  );
  expect(finishResidentJoinCompositeHash(12345)).toBeLessThanOrEqual(0xffffffff);
});

test('exact string composite keys equal a nested-loop join for every join type', () => {
  for (let seed = 1; seed <= 6; seed++) {
    const random = seededRandom(seed);
    const left = [
      makeStringColumn(random, ['a', 'b', 'c', '', 'd'], 40),
      makeStringColumn(random, ['x', 'y', 'z'], 40)
    ];
    const right = [
      makeStringColumn(random, ['b', 'a', 'e', 'c', ''], 30),
      makeStringColumn(random, ['z', 'x', 'q'], 30)
    ];
    const keyPlan = makeKeyPlan(left, right, 'sint32');
    expect(keyPlan.exact).toBe(true);
    for (const joinType of ['INNER', 'LEFT', 'FULL'] as const)
      expect(
        mirrorPairs({plan: keyPlan, joinType, left, right, format: 'sint32'}),
        `${seed} ${joinType}`
      ).toEqual(nestedLoopPairs(joinType, left, right));
  }
});

test('hashed numeric and mixed keys match the oracle once verified, even with all buckets colliding', () => {
  for (let seed = 11; seed <= 16; seed++) {
    const random = seededRandom(seed);
    const left = [
      makeStringColumn(random, ['a', 'b', 'c'], 36),
      makeNumberColumn(random, 36, 4),
      makeNumberColumn(random, 36, 3)
    ];
    const right = [
      makeStringColumn(random, ['c', 'a', 'b', 'e'], 28),
      makeNumberColumn(random, 28, 4),
      makeNumberColumn(random, 28, 3)
    ];
    for (const format of ['sint32', 'float32'] as const) {
      const keyPlan = makeKeyPlan(left, right, format);
      expect(keyPlan.exact).toBe(false);
      const expected = nestedLoopPairs('INNER', left, right);
      expect(mirrorPairs({plan: keyPlan, joinType: 'INNER', left, right, format})).toEqual(
        expected
      );
      // One bucket makes every numeric combination collide; the strings stay exact.
      const colliding = {...keyPlan, hashModulus: 1};
      expect(
        mirrorPairs({plan: colliding, joinType: 'INNER', left, right, format}),
        `collide ${seed} ${format}`
      ).toEqual(expected);
    }
  }
  const random = seededRandom(99);
  const leftNumbers = [makeNumberColumn(random, 30, 5), makeNumberColumn(random, 30, 5)];
  const rightNumbers = [makeNumberColumn(random, 25, 5), makeNumberColumn(random, 25, 5)];
  const keyPlan = makeKeyPlan(leftNumbers, rightNumbers, 'sint32');
  expect(
    mirrorPairs({
      plan: keyPlan,
      joinType: 'INNER',
      left: leftNumbers,
      right: rightNumbers,
      format: 'sint32'
    })
  ).toEqual(nestedLoopPairs('INNER', leftNumbers, rightNumbers));
  expect(
    mirrorPairs({
      plan: {...keyPlan, hashModulus: 1},
      joinType: 'INNER',
      left: leftNumbers,
      right: rightNumbers,
      format: 'sint32'
    })
  ).toEqual(nestedLoopPairs('INNER', leftNumbers, rightNumbers));
});

test('SEMI is INNER with the first match and ANTI keeps unmatched left rows (NULL keys included)', () => {
  for (let seed = 21; seed <= 25; seed++) {
    const random = seededRandom(seed);
    const left = [
      makeStringColumn(random, ['a', 'b', 'c', 'd'], 32),
      makeStringColumn(random, ['x', 'y'], 32)
    ];
    const right = [
      makeStringColumn(random, ['b', 'a', 'e'], 20),
      makeStringColumn(random, ['y', 'x'], 20)
    ];
    const keyPlan = makeKeyPlan(left, right, 'sint32');
    const oracle = nestedLoopPairs('LEFT', left, right);
    const matchedLeftRows = new Set(
      oracle.filter(pair => !pair.endsWith(':-')).map(pair => pair.split(':')[0])
    );
    const semi = mirrorPairs({
      plan: keyPlan,
      joinType: 'INNER',
      left,
      right,
      format: 'sint32',
      duplicateRightMatches: 'first'
    }).map(pair => pair.split(':')[0]);
    expect(new Set(semi)).toEqual(matchedLeftRows);
    expect(semi.length).toBe(matchedLeftRows.size);
    const anti = mirrorPairs({
      plan: keyPlan,
      joinType: 'LEFT',
      left,
      right,
      format: 'sint32',
      duplicateRightMatches: 'first'
    })
      .filter(pair => pair.endsWith(':-'))
      .map(pair => pair.split(':')[0]);
    const expectedAnti = Array.from({length: 32}, (_, row) => String(row)).filter(
      row => !matchedLeftRows.has(row)
    );
    expect([...anti].sort()).toEqual([...expectedAnti].sort());
    // Rows whose key has a NULL never match, so ANTI keeps them.
    const nullRows = left[0].values
      .map((value, row) => (value === null || left[1].values[row] === null ? String(row) : ''))
      .filter(Boolean);
    for (const row of nullRows) expect(anti).toContain(row);
  }
});

test('string key spaces that cannot fold into 32 bits are refused', () => {
  const big = Array.from({length: 20_000}, (_, index) => `v${index}`);
  const left = [
    {kind: 'string', dictionary: big, values: ['v1']} as KeyColumn,
    {kind: 'string', dictionary: big, values: ['v2']} as KeyColumn,
    {kind: 'string', dictionary: big, values: ['v3']} as KeyColumn
  ];
  expect(() => makeKeyPlan(left, left, 'sint32')).toThrow(/too many label combinations/);
});

/** Sorted distinct canonical key words of both sides for each numeric component (test oracle). */
function makeRankKeys(
  plan: ResidentJoinCompositeKeyPlan,
  left: KeyColumn[],
  right: KeyColumn[],
  format: 'sint32' | 'float32'
): (Uint32Array | undefined)[] {
  return plan.components.map((component, index) => {
    if (component.kind !== 'numeric') return undefined;
    const keys = new Set<number>();
    for (const column of [left[index], right[index]])
      for (const value of column.values)
        if (value !== null) keys.add(new Int32Array([value as number])[0] >>> 0);
    void format;
    return Uint32Array.from([...keys].sort((a, b) => a - b));
  });
}

test('ranked numeric composite keys are exact for every join type and duplicate policy', () => {
  for (let seed = 31; seed <= 38; seed++) {
    const random = seededRandom(seed);
    const left = [
      makeNumberColumn(random, 36, 5),
      makeStringColumn(random, ['a', 'b', 'c'], 36),
      makeNumberColumn(random, 36, 4)
    ];
    const right = [
      makeNumberColumn(random, 27, 5),
      makeStringColumn(random, ['c', 'a', 'e'], 27),
      makeNumberColumn(random, 27, 4)
    ];
    for (const format of ['sint32', 'float32'] as const) {
      const hashed = makeKeyPlan(left, right, format);
      expect(hashed.exact).toBe(false);
      const ranked = makeResidentJoinRankedCompositeKeyPlan(
        hashed,
        makeRankKeys(hashed, left, right, format)
      );
      expect(ranked?.exact).toBe(true);
      for (const joinType of ['INNER', 'LEFT', 'FULL'] as const)
        expect(
          mirrorPairs({plan: ranked!, joinType, left, right, format}),
          `${seed} ${format} ${joinType}`
        ).toEqual(nestedLoopPairs(joinType, left, right));
      // SEMI (first match) keeps each matching left row once; ANTI keeps the unmatched rows.
      const oracle = nestedLoopPairs('LEFT', left, right);
      const matchedLeft = new Set(
        oracle.filter(pair => !pair.endsWith(':-')).map(pair => pair.split(':')[0])
      );
      const semi = mirrorPairs({
        plan: ranked!,
        joinType: 'INNER',
        left,
        right,
        format,
        duplicateRightMatches: 'first'
      }).map(pair => pair.split(':')[0]);
      expect(new Set(semi)).toEqual(matchedLeft);
      expect(semi.length).toBe(matchedLeft.size);
      const anti = mirrorPairs({
        plan: ranked!,
        joinType: 'LEFT',
        left,
        right,
        format,
        duplicateRightMatches: 'first'
      })
        .filter(pair => pair.endsWith(':-'))
        .map(pair => pair.split(':')[0]);
      expect(new Set(anti)).toEqual(
        new Set(oracle.filter(pair => pair.endsWith(':-')).map(pair => pair.split(':')[0]))
      );
    }
  }
});

test('ranking refuses a mixed radix product that no longer fits one 32-bit word', () => {
  const left = [makeNumberColumn(seededRandom(5), 8, 3), makeNumberColumn(seededRandom(6), 8, 3)];
  const right = [makeNumberColumn(seededRandom(7), 8, 3), makeNumberColumn(seededRandom(8), 8, 3)];
  const hashed = makeKeyPlan(left, right, 'sint32');
  const huge = Uint32Array.from({length: 70_000}, (_, index) => index);
  expect(makeResidentJoinRankedCompositeKeyPlan(hashed, [huge, huge])).toBeUndefined();
  expect(
    makeResidentJoinRankedCompositeKeyPlan(hashed, [huge, Uint32Array.from([1, 2])])
  ).toBeDefined();
});
