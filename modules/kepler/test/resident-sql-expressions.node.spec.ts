// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  formatResidentSqlFloat,
  getResidentSqlNumericExpressionBody,
  mirrorResidentSqlNumericExpression,
  parseResidentSqlEpochSeconds,
  parseResidentSqlExpression,
  planResidentSqlNumericExpression
} from '../../../examples/deck/kepler-compute/src/resident-sql-expressions';
import {
  makeResidentSqlOrderedKey,
  mirrorResidentSqlCastKernel,
  readResidentSqlOrderedKey
} from '../../../examples/deck/kepler-compute/src/resident-sql-cast-kernel';
import {makeResidentSqlValueDictionary} from '../../../examples/deck/kepler-compute/src/resident-sql-distinct';
import {
  rewriteResidentSql,
  rewriteResidentSqlBetween,
  tokenizeResidentSql,
  type ResidentSqlDerivedColumn,
  type ResidentSqlTableInfo
} from '../../../examples/deck/kepler-compute/src/resident-sql-strings';

const NAMES = [' Bob ', 'alice', 'Carol', 'carol', '', 'dave smith', 'Zoë'];
const CITIES = ['Paris', 'Rome', 'Oslo'];
const table: ResidentSqlTableInfo = {
  name: 'people',
  columns: {
    name: {format: 'uint32', dictionary: NAMES},
    city: {format: 'uint32', dictionary: CITIES},
    other: {format: 'uint32', dictionary: ['Rome', 'Oslo', 'Zurich']},
    score: {format: 'float32'},
    age: {format: 'sint32'},
    code: {format: 'uint32'}
  }
};
const find = (name: string) => (name === 'people' ? table : undefined);
const gather = (derived: ResidentSqlDerivedColumn, codes: number[]) => {
  if (derived.kind !== 'lookup') throw new Error('lookup expected');
  return codes.map(code => derived.words[code]);
};
const labelsOf = (derived: ResidentSqlDerivedColumn, codes: number[]) =>
  gather(derived, codes).map(word => (derived as {dictionary: string[]}).dictionary[word]);
const ALL_NAMES = NAMES.map((_, code) => code);

test('BETWEEN becomes two comparisons, NOT BETWEEN two alternatives', () => {
  expect(rewriteResidentSqlBetween('SELECT a FROM t WHERE a BETWEEN 1 AND 5')).toBe(
    'SELECT a FROM t WHERE (a >= 1 AND a <= 5)'
  );
  expect(
    rewriteResidentSqlBetween('SELECT a FROM t WHERE b > 1 AND a + 1 NOT BETWEEN 2 AND 9 OR c = 2')
  ).toBe('SELECT a FROM t WHERE b > 1 AND ((a + 1) < 2 OR (a + 1) > 9) OR c = 2');
  expect(
    rewriteResidentSqlBetween(
      "SELECT a FROM t WHERE (a BETWEEN 1 AND 2) AND n BETWEEN 'a' AND 'm' ORDER BY a"
    )
  ).toBe("SELECT a FROM t WHERE ((a >= 1 AND a <= 2)) AND (n >= 'a' AND n <= 'm') ORDER BY a");
  expect(() => rewriteResidentSqlBetween('SELECT a FROM t WHERE a BETWEEN 1')).toThrow(/BETWEEN/);
  // BETWEEN over a string column reuses the string predicate masks.
  const rewrite = rewriteResidentSql(
    "SELECT age FROM people WHERE city BETWEEN 'Oslo' AND 'Rome'",
    find
  )!;
  expect(rewrite.derived).toHaveLength(2);
  const [low, high] = rewrite.derived.map(spec => gather(spec, [0, 1, 2]));
  expect(low).toEqual([1, 1, 1]);
  expect(high).toEqual(
    [1, 1, 0].map((_, index) => (['Paris', 'Rome', 'Oslo'][index] <= 'Rome' ? 1 : 0))
  );
});

test('string functions evaluate per dictionary label like the SQL definitions', () => {
  const run = (expression: string) => {
    const rewrite = rewriteResidentSql(`SELECT ${expression} AS r FROM people`, find)!;
    expect(rewrite.sql).toBe('SELECT r FROM people');
    return rewrite.derived[0];
  };
  expect(labelsOf(run('UPPER(name)'), ALL_NAMES)).toEqual(NAMES.map(name => name.toUpperCase()));
  expect(labelsOf(run('LOWER(name)'), ALL_NAMES)).toEqual(NAMES.map(name => name.toLowerCase()));
  // TRIM removes spaces only, LENGTH counts characters.
  expect(labelsOf(run('TRIM(name)'), ALL_NAMES)).toEqual(
    NAMES.map(name => name.replace(/^ +| +$/g, ''))
  );
  expect(gather(run('LENGTH(name)'), ALL_NAMES)).toEqual(
    NAMES.map(name => Array.from(name).length)
  );
  expect(labelsOf(run('SUBSTR(name, 2, 3)'), ALL_NAMES)).toEqual(
    NAMES.map(name => Array.from(name).slice(1, 4).join(''))
  );
  expect(labelsOf(run("REPLACE(name, 'a', '*')"), ALL_NAMES)).toEqual(
    NAMES.map(name => name.split('a').join('*'))
  );
  expect(labelsOf(run("UPPER(TRIM(name)) || '!'"), ALL_NAMES)).toEqual(
    NAMES.map(name => `${name.replace(/^ +| +$/g, '').toUpperCase()}!`)
  );
  expect(labelsOf(run("'<' || name || '>'"), ALL_NAMES)).toEqual(NAMES.map(name => `<${name}>`));
  // The result dictionary is deduplicated and sorted, so 'Carol' and 'carol' share a code under LOWER.
  const lower = run('LOWER(name)') as {dictionary: string[]; words: Uint32Array};
  expect(new Set(lower.dictionary).size).toBe(lower.dictionary.length);
  expect(lower.words[2]).toBe(lower.words[3]);
  expect([...lower.dictionary]).toEqual([...lower.dictionary].sort());
});

test('string function results feed predicates, grouping and deduplicate repeated expressions', () => {
  const rewrite = rewriteResidentSql(
    "SELECT UPPER(name) AS u, COUNT(*) AS n FROM people WHERE UPPER(name) = 'CAROL' AND LENGTH(name) > 3 GROUP BY UPPER(name)",
    find
  )!;
  const names = rewrite.derived.map(spec => spec.name);
  expect(names[0]).toBe('u');
  expect(rewrite.derived.filter(spec => spec.kind === 'lookup' && spec.dictionary)).toHaveLength(1);
  expect(rewrite.sql).toMatch(
    /^SELECT u, COUNT\(\*\) AS n FROM people WHERE \(__luma_sql_mask_\d+ = 1\) AND __luma_sql_expr_\d+ > 3 GROUP BY u$/
  );
  const mask = rewrite.derived.find(spec => spec.name.startsWith('__luma_sql_mask'))!;
  const upper = rewrite.derived[0] as {words: Uint32Array; dictionary: string[]};
  // mask is evaluated over the codes of u (the new dictionary).
  expect(rewrite.derived.indexOf(mask)).toBeGreaterThan(0);
  expect((mask as {sourceColumn: string}).sourceColumn).toBe('u');
  expect(upper.dictionary).toContain('CAROL');
  expect(
    gather(
      mask,
      upper.dictionary.map((_, code) => code)
    )
  ).toEqual(upper.dictionary.map(label => (label === 'CAROL' ? 1 : 0)));
});

test('two-column string expressions are bounded by the label product', () => {
  const pair = rewriteResidentSql("SELECT city || '-' || other AS route FROM people", find)!;
  const spec = pair.derived[0] as Extract<ResidentSqlDerivedColumn, {kind: 'pairLookup'}>;
  expect(spec).toMatchObject({kind: 'pairLookup', firstSize: 3, secondSize: 3});
  const pick = (a: number, b: number) => spec.dictionary![spec.words[a * 3 + b]];
  expect(pick(1, 0)).toBe('Rome-Rome');
  expect(pick(2, 2)).toBe('Oslo-Zurich');
  const big: ResidentSqlTableInfo = {
    name: 'big',
    columns: {
      a: {format: 'uint32', dictionary: Array.from({length: 300}, (_, i) => `a${i}`)},
      b: {format: 'uint32', dictionary: Array.from({length: 300}, (_, i) => `b${i}`)},
      c: {format: 'uint32', dictionary: ['x']}
    }
  };
  expect(() =>
    rewriteResidentSql('SELECT a || b AS ab FROM big', name => (name === 'big' ? big : undefined))
  ).toThrow(/unbounded/);
  expect(() =>
    rewriteResidentSql('SELECT a || b || c AS abc FROM big', name =>
      name === 'big' ? big : undefined
    )
  ).toThrow(/unbounded/);
  expect(() => rewriteResidentSql('SELECT UPPER(age) FROM people', find)).toThrow();
});

test('comparing two string columns compares ranks in the shared dictionary', () => {
  const rewrite = rewriteResidentSql('SELECT age FROM people WHERE city = other', find)!;
  expect(rewrite.derived).toHaveLength(2);
  const [left, right] = rewrite.derived.map(spec => (spec.kind === 'lookup' ? spec.words : null));
  const equal = [0, 1, 2]
    .flatMap(a => [0, 1, 2].map(b => [a, b]))
    .filter(([a, b]) => left![a] === right![b]);
  const oracle = [0, 1, 2]
    .flatMap(a => [0, 1, 2].map(b => [a, b]))
    .filter(([a, b]) => CITIES[a] === ['Rome', 'Oslo', 'Zurich'][b]);
  expect(equal).toEqual(oracle);
  const less = rewriteResidentSql('SELECT age FROM people WHERE city < other', find)!;
  const [lessLeft, lessRight] = less.derived.map(spec =>
    spec.kind === 'lookup' ? spec.words : null
  );
  for (const a of [0, 1, 2])
    for (const b of [0, 1, 2])
      expect(lessLeft![a] < lessRight![b]).toBe(CITIES[a] < ['Rome', 'Oslo', 'Zurich'][b]);
  expect(less.sql).toMatch(/\(__luma_sql_mask_\d+ < __luma_sql_mask_\d+\)/);
});

test('numeric CAST expressions: kernel body and CPU mirror agree with SQL arithmetic', () => {
  const columns = {score: 'float32', age: 'sint32', code: 'uint32'} as const;
  const parse = (text: string) => {
    const tokens = tokenizeResidentSql(text);
    return parseResidentSqlExpression(tokens, 0, tokens.length, token => token.text);
  };
  const plan = (text: string) =>
    planResidentSqlNumericExpression(parse(text), name => columns[name as keyof typeof columns]);
  const run = (text: string, words: Record<string, number>, validity = {}) => {
    const planned = plan(text);
    return mirrorResidentSqlNumericExpression(planned, words, validity);
  };
  const f32 = (value: number) => new Uint32Array(new Float32Array([value]).buffer)[0];
  const i32 = (value: number) => value >>> 0;
  expect(plan('age + age * 2').resultFormat).toBe('sint32');
  expect(plan('age / 2').resultFormat).toBe('float32');
  expect(plan('score + age').resultFormat).toBe('float32');
  const sum = run('age + age * 2', {age: i32(7)});
  expect(mirrorResidentSqlCastKernel('sint32', 'sint32', sum.word)).toEqual({
    word: 21,
    valid: true
  });
  const ratio = run('age / 2', {age: i32(7)});
  expect(mirrorResidentSqlCastKernel('float32', 'sint32', ratio.word).word).toBe(4); // 3.5 rounds away from zero
  expect(run('age / (age - age)', {age: i32(5)}).valid).toBe(false);
  expect(run('SQRT(score)', {score: f32(-1)}).valid).toBe(false);
  expect(
    mirrorResidentSqlCastKernel(
      'float32',
      'sint32',
      run('ROUND(score * 2)', {score: f32(1.3)}).word
    ).word
  ).toBe(3);
  expect(run('-age', {age: i32(-4)}).word).toBe(4);
  expect(run('code + 1', {code: 3_000_000_000}).valid).toBe(false);
  expect(run('age', {age: i32(1)}, {age: false}).valid).toBe(false);
  const body = getResidentSqlNumericExpressionBody(plan('age / 2 + score'), [true, false]);
  expect(body).toContain('validityIn0[index]');
  expect(body).toContain('let word = bitcast<u32>');
  expect(() => plan('UPPER(name)')).toThrow();
});

test('DATE and TIMESTAMP casts are epoch-second math', () => {
  expect(parseResidentSqlEpochSeconds('2024-02-29', 'date')).toBe(Date.UTC(2024, 1, 29) / 1000);
  expect(parseResidentSqlEpochSeconds('2024-02-29 13:45:10', 'timestamp')).toBe(
    Date.UTC(2024, 1, 29, 13, 45, 10) / 1000
  );
  expect(parseResidentSqlEpochSeconds('2024-02-29T13:45:10Z', 'date')).toBe(
    Date.UTC(2024, 1, 29) / 1000
  );
  expect(() => parseResidentSqlEpochSeconds('2023-02-29', 'date')).toThrow(/would fail/);
  expect(() => parseResidentSqlEpochSeconds('2024-01-01 00:00:00.5', 'timestamp')).toThrow(
    /fractional/
  );
  expect(() => parseResidentSqlEpochSeconds('3000-01-01', 'date')).toThrow(/range/);
  // Numeric epoch seconds: TIMESTAMP keeps the second, DATE floors to UTC midnight (also negative).
  const seconds = 1_700_000_123;
  expect(mirrorResidentSqlCastKernel('sint32', 'timestamp', seconds)).toEqual({
    word: seconds,
    valid: true
  });
  expect(mirrorResidentSqlCastKernel('sint32', 'date', seconds).word).toBe(
    Math.floor(seconds / 86400) * 86400
  );
  expect(mirrorResidentSqlCastKernel('sint32', 'date', -1 >>> 0).word | 0).toBe(-86400);
  expect(mirrorResidentSqlCastKernel('uint32', 'timestamp', 4_000_000_000).valid).toBe(false);
  expect(() => rewriteResidentSql("SELECT CAST('x' AS DATE) FROM people", find)).toThrow(
    /constant string/
  );
});

test('date/timestamp casts carry Kepler time metadata and string labels parse once per label', () => {
  const dates: ResidentSqlTableInfo = {
    name: 'events',
    columns: {
      day: {format: 'uint32', dictionary: ['2024-01-01', '2024-01-02 03:04:05', '1969-12-31']},
      epoch: {format: 'sint32'}
    }
  };
  const lookup = (name: string) => (name === 'events' ? dates : undefined);
  const fromString = rewriteResidentSql('SELECT CAST(day AS DATE) AS d FROM events', lookup)!;
  expect([...(fromString.derived[0] as {words: Uint32Array}).words].map(word => word | 0)).toEqual([
    Date.UTC(2024, 0, 1) / 1000,
    Date.UTC(2024, 0, 2) / 1000,
    -86400
  ]);
  expect(fromString.columnMetadata.d).toMatchObject({
    'luma.gl.time.format': 'X',
    'luma.gl.sql.cast': 'DATE'
  });
  const fromNumber = rewriteResidentSql(
    'SELECT CAST(epoch AS TIMESTAMP) AS t FROM events',
    lookup
  )!;
  expect(fromNumber.derived[0]).toMatchObject({kind: 'cast', target: 'timestamp'});
  expect(fromNumber.columnMetadata.t['luma.gl.sql.cast']).toBe('TIMESTAMP');
  const expression = rewriteResidentSql(
    'SELECT CAST(epoch * 60 AS TIMESTAMP) AS t FROM events',
    lookup
  )!;
  expect(expression.derived[0]).toMatchObject({kind: 'expression', target: 'timestamp'});
});

test('numeric to VARCHAR builds a sorted dictionary from distinct keys', () => {
  expect(formatResidentSqlFloat(1.5)).toBe('1.5');
  expect(formatResidentSqlFloat(3)).toBe('3.0');
  expect(formatResidentSqlFloat(Math.fround(0.1))).toBe('0.1');
  expect(formatResidentSqlFloat(-2.25)).toBe('-2.25');
  expect(formatResidentSqlFloat(1e20)).toBe('1e+20');
  const floats = [-3.5, 0, 2, 10.25, 2];
  const words = floats.map(value => new Uint32Array(new Float32Array([value]).buffer)[0]);
  const keys = words.map(word => makeResidentSqlOrderedKey('float32', word));
  expect(
    [...keys].sort((a, b) => a - b).map(key => readResidentSqlOrderedKey('float32', key))
  ).toEqual(
    [...words].sort(
      (a, b) =>
        new Float32Array(new Uint32Array([a]).buffer)[0] -
        new Float32Array(new Uint32Array([b]).buffer)[0]
    )
  );
  const sortedUnique = [...new Set(keys)].sort((a, b) => a - b);
  const dictionary = makeResidentSqlValueDictionary('float32', sortedUnique);
  expect([...dictionary.labels]).toEqual(['-3.5', '0.0', '10.25', '2.0']);
  // Each key's code names its label.
  expect([...dictionary.codes].map(code => dictionary.labels[code])).toEqual([
    '-3.5',
    '0.0',
    '2.0',
    '10.25'
  ]);
  const integers = makeResidentSqlValueDictionary(
    'sint32',
    [-5, 12, 7].map(value => makeResidentSqlOrderedKey('sint32', value >>> 0)).sort((a, b) => a - b)
  );
  expect([...integers.codes].map(code => integers.labels[code])).toEqual(['-5', '7', '12']);
  // Resolved dictionaries make the cast an ordinary string column for predicates.
  const resolved = rewriteResidentSql(
    "SELECT age FROM people WHERE CAST(code AS VARCHAR) = '12'",
    find,
    {
      resolveValueDictionary: () => integers
    }
  )!;
  expect(resolved.pendingValueColumns).toEqual([]);
  expect(resolved.derived.map(spec => spec.kind)).toEqual(['valueLookup', 'lookup']);
  const mask = resolved.derived[1] as {words: Uint32Array};
  expect([...mask.words]).toEqual(integers.labels.map(label => (label === '12' ? 1 : 0)));
});
