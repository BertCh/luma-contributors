// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  castResidentSqlValue,
  classifyResidentSqlLikePattern,
  compareResidentSqlStrings,
  isResidentSqlDictionarySorted,
  makeResidentSqlRankLookup,
  makeResidentSqlStringCastLookup,
  RESIDENT_SQL_HIDDEN_PREFIX,
  rewriteResidentSql,
  type ResidentSqlDerivedColumn,
  type ResidentSqlScalarFormat,
  type ResidentSqlTableInfo
} from '../../../examples/deck/kepler-compute/src/resident-sql-strings';
import {
  getResidentSqlCastKernelBody,
  mirrorResidentSqlCastKernel
} from '../../../examples/deck/kepler-compute/src/resident-sql-cast-kernel';

const CITIES = ['Zurich', 'Alpha', 'Beta', 'alphabet', 'Be_ta', '', 'Zürich'];
const table: ResidentSqlTableInfo = {
  name: 'places',
  columns: {
    city: {format: 'uint32', dictionary: CITIES},
    sorted: {format: 'uint32', dictionary: ['a', 'b', 'c']},
    amount: {format: 'float32'},
    count: {format: 'sint32'},
    id: {format: 'uint32'},
    digits: {format: 'uint32', dictionary: ['1', '22', '3.0']}
  }
};
const find = (name: string) => (name === 'places' ? table : undefined);

/** Independent LIKE oracle: dynamic programming, no regular expressions. */
function likeOracle(text: string, pattern: string, escapeCharacter?: string): boolean {
  const t = Array.from(text);
  const tokens: {kind: 'any' | 'one' | 'char'; value?: string}[] = [];
  const p = Array.from(pattern);
  for (let index = 0; index < p.length; index++) {
    if (escapeCharacter !== undefined && p[index] === escapeCharacter && index + 1 < p.length)
      tokens.push({kind: 'char', value: p[++index]});
    else if (p[index] === '%') tokens.push({kind: 'any'});
    else if (p[index] === '_') tokens.push({kind: 'one'});
    else tokens.push({kind: 'char', value: p[index]});
  }
  const memo = new Map<string, boolean>();
  const go = (i: number, j: number): boolean => {
    const key = `${i},${j}`;
    if (memo.has(key)) return memo.get(key)!;
    let result: boolean;
    if (j === tokens.length) result = i === t.length;
    else if (tokens[j].kind === 'any') result = go(i, j + 1) || (i < t.length && go(i + 1, j));
    else if (i === t.length) result = false;
    else result = (tokens[j].kind === 'one' || tokens[j].value === t[i]) && go(i + 1, j + 1);
    memo.set(key, result);
    return result;
  };
  return go(0, 0);
}

/** What the GPU does: gather the per-label lookup through the dictionary codes. */
function evaluateMask(derived: ResidentSqlDerivedColumn, codes: number[]): number[] {
  if (derived.kind !== 'lookup') throw new Error('lookup expected');
  return codes.map(code => (code < derived.words.length ? derived.words[code] : 0));
}
const ALL_CODES = CITIES.map((_, code) => code);

test('string equality and inequality become dictionary masks matching an oracle', () => {
  const equal = rewriteResidentSql("SELECT id FROM places WHERE city = 'Beta'", find)!;
  expect(equal.sql).toBe(`SELECT id FROM places WHERE (${equal.derived[0].name} = 1)`);
  expect(equal.derived[0].name.startsWith(RESIDENT_SQL_HIDDEN_PREFIX)).toBe(true);
  expect(evaluateMask(equal.derived[0], ALL_CODES)).toEqual(
    CITIES.map(city => (city === 'Beta' ? 1 : 0))
  );
  const notEqual = rewriteResidentSql("SELECT id FROM places WHERE 'Beta' <> city", find)!;
  expect(evaluateMask(notEqual.derived[0], ALL_CODES)).toEqual(
    CITIES.map(city => (city !== 'Beta' ? 1 : 0))
  );
  const greater = rewriteResidentSql("SELECT id FROM places WHERE city >= 'Beta'", find)!;
  expect(evaluateMask(greater.derived[0], ALL_CODES)).toEqual(
    CITIES.map(city => (compareResidentSqlStrings(city, 'Beta') >= 0 ? 1 : 0))
  );
  const flipped = rewriteResidentSql("SELECT id FROM places WHERE 'Beta' < city", find)!;
  expect(evaluateMask(flipped.derived[0], ALL_CODES)).toEqual(
    CITIES.map(city => (compareResidentSqlStrings(city, 'Beta') > 0 ? 1 : 0))
  );
  // A literal missing from the dictionary matches nothing.
  const missing = rewriteResidentSql("SELECT id FROM places WHERE city = 'Nowhere'", find)!;
  expect(evaluateMask(missing.derived[0], ALL_CODES)).toEqual(CITIES.map(() => 0));
});

test('IN and NOT IN lists', () => {
  const inList = rewriteResidentSql(
    "SELECT id FROM places WHERE city IN ('Beta', 'Alpha', '')",
    find
  )!;
  expect(evaluateMask(inList.derived[0], ALL_CODES)).toEqual(
    CITIES.map(city => (['Beta', 'Alpha', ''].includes(city) ? 1 : 0))
  );
  const notIn = rewriteResidentSql("SELECT id FROM places WHERE city NOT IN ('Beta')", find)!;
  expect(evaluateMask(notIn.derived[0], ALL_CODES)).toEqual(
    CITIES.map(city => (city !== 'Beta' ? 1 : 0))
  );
  expect(() => rewriteResidentSql("SELECT id FROM places WHERE city IN ('a', 1)", find)).toThrow(
    /string literals only/
  );
});

test('LIKE and ILIKE prefix, suffix, contains and general patterns agree with the oracle', () => {
  const cases: {pattern: string; ilike?: boolean; escapeCharacter?: string; kind: string}[] = [
    {pattern: 'Be%', kind: 'prefix'},
    {pattern: '%ta', kind: 'suffix'},
    {pattern: '%lph%', kind: 'contains'},
    {pattern: 'Beta', kind: 'equals'},
    {pattern: 'Be_ta', kind: 'general'},
    {pattern: 'Z_rich', kind: 'general'},
    {pattern: '%a%a%', kind: 'general'},
    {pattern: 'be%', ilike: true, kind: 'prefix'},
    {pattern: 'ALPHA%', ilike: true, kind: 'prefix'},
    {pattern: 'Be\\_ta', escapeCharacter: '\\', kind: 'general'},
    {pattern: '%', kind: 'suffix'},
    {pattern: '', kind: 'equals'}
  ];
  for (const {pattern, ilike, escapeCharacter, kind} of cases) {
    expect(classifyResidentSqlLikePattern(pattern, escapeCharacter).kind).toBe(kind);
    const sql = `SELECT id FROM places WHERE city ${ilike ? 'ILIKE' : 'LIKE'} '${pattern.replaceAll("'", "''")}'${escapeCharacter ? ` ESCAPE '${escapeCharacter}'` : ''}`;
    const rewritten = rewriteResidentSql(sql, find)!;
    const fold = (text: string) => (ilike ? text.toLowerCase() : text);
    expect(evaluateMask(rewritten.derived[0], ALL_CODES), sql).toEqual(
      CITIES.map(city =>
        likeOracle(fold(city), fold(pattern), escapeCharacter && fold(escapeCharacter)) ? 1 : 0
      )
    );
    const negated = rewriteResidentSql(
      sql.replace(' LIKE ', ' NOT LIKE ').replace(' ILIKE ', ' NOT ILIKE '),
      find
    )!;
    expect(evaluateMask(negated.derived[0], ALL_CODES), sql).toEqual(
      CITIES.map(city =>
        likeOracle(fold(city), fold(pattern), escapeCharacter && fold(escapeCharacter)) ? 0 : 1
      )
    );
  }
});

test('predicates combine and keep the rest of the statement', () => {
  const rewritten = rewriteResidentSql(
    "SELECT id, amount FROM places WHERE city LIKE 'Be%' AND amount > 2 OR city = 'Zurich' ORDER BY amount LIMIT 3",
    find
  )!;
  expect(rewritten.derived).toHaveLength(2);
  expect(rewritten.sql).toBe(
    `SELECT id, amount FROM places WHERE (${rewritten.derived[0].name} = 1) AND amount > 2 OR (${rewritten.derived[1].name} = 1) ORDER BY amount LIMIT 3`
  );
});

test('string columns project by zero-copy alias and star does not leak helper columns', () => {
  const alias = rewriteResidentSql('SELECT city AS town, amount FROM places', find)!;
  expect(alias.sql).toBe('SELECT town, amount FROM places');
  expect(alias.derived).toEqual([
    {kind: 'alias', name: 'town', hidden: false, sourceColumn: 'city'}
  ]);
  expect(() => rewriteResidentSql('SELECT city AS id FROM places', find)).toThrow(/collides/);
  // A numeric column alias needs no rewrite.
  expect(rewriteResidentSql('SELECT amount AS a FROM places', find)).toBeNull();
  const star = rewriteResidentSql("SELECT * FROM places WHERE city = 'Beta'", find)!;
  expect(star.sql).toContain('SELECT city, sorted, amount, count, id, digits FROM places');
});

test('ORDER BY a string sorts by label rank unless the dictionary is already ordered', () => {
  expect(isResidentSqlDictionarySorted(table.columns['sorted'].dictionary!)).toBe(true);
  expect(rewriteResidentSql('SELECT sorted FROM places ORDER BY sorted', find)).toBeNull();
  const ordered = rewriteResidentSql('SELECT city, id FROM places ORDER BY city DESC', find)!;
  const rank = ordered.derived[0];
  expect(rank.name.startsWith(RESIDENT_SQL_HIDDEN_PREFIX)).toBe(true);
  expect(ordered.sql).toBe(`SELECT city, id, ${rank.name} FROM places ORDER BY ${rank.name} DESC`);
  // Sorting codes by rank equals sorting labels in code point order.
  const ranks = evaluateMask(rank, ALL_CODES);
  const byRank = [...ALL_CODES].sort((a, b) => ranks[a] - ranks[b] || a - b);
  const byLabel = [...ALL_CODES].sort(
    (a, b) => compareResidentSqlStrings(CITIES[a], CITIES[b]) || a - b
  );
  expect(byRank.map(code => CITIES[code])).toEqual(byLabel.map(code => CITIES[code]));
  // Ordering through a select alias resolves to the source dictionary.
  const aliased = rewriteResidentSql('SELECT city AS town FROM places ORDER BY town', find)!;
  expect(aliased.derived.map(entry => entry.kind)).toEqual(['alias', 'lookup']);
  expect(makeResidentSqlRankLookup(['b', 'a', 'b'])).toEqual(Uint32Array.of(1, 0, 1).fill(1, 0, 1));
});

test('unsupported string uses refuse with an explicit reason', () => {
  expect(() => rewriteResidentSql("SELECT id FROM places WHERE amount = 'x'", find)).toThrow(
    /String literal outside a supported predicate/
  );
  expect(() => rewriteResidentSql("SELECT 'x' AS label FROM places", find)).toThrow(
    /String literal outside/
  );
  expect(rewriteResidentSql('SELECT id FROM places WHERE id > 3', find)).toBeNull();
  expect(rewriteResidentSql('SELECT id FROM unknown WHERE id > 3', find)).toBeNull();
});

test('CAST among numeric types, string passthrough and refusals', () => {
  const toFloat = rewriteResidentSql('SELECT CAST(count AS DOUBLE) AS count_f FROM places', find)!;
  expect(toFloat.sql).toBe('SELECT count_f FROM places');
  expect(toFloat.derived).toEqual([
    {
      kind: 'cast',
      name: 'count_f',
      hidden: false,
      sourceColumn: 'count',
      sourceFormat: 'sint32',
      target: 'float32'
    }
  ]);
  const inline = rewriteResidentSql('SELECT id FROM places WHERE amount::INTEGER = 3', find)!;
  expect(inline.derived[0]).toMatchObject({kind: 'cast', hidden: true, target: 'sint32'});
  expect(inline.sql).toBe(`SELECT id FROM places WHERE ${inline.derived[0].name} = 3`);
  const noAlias = rewriteResidentSql('SELECT CAST(amount AS BIGINT) FROM places', find)!;
  expect(noAlias.derived[0].name).toBe('cast_amount');
  expect(noAlias.sql).toBe('SELECT cast_amount FROM places');
  const bool = rewriteResidentSql('SELECT amount::BOOLEAN AS flag FROM places', find)!;
  expect(bool.derived[0]).toMatchObject({target: 'bool', name: 'flag'});
  // Same-type casts are identities.
  expect(
    rewriteResidentSql('SELECT CAST(amount AS REAL) AS a2 FROM places', find)!.derived
  ).toEqual([]);
  // Strings: a dictionary column to VARCHAR is the column; numeric to VARCHAR is refused.
  expect(rewriteResidentSql('SELECT CAST(city AS VARCHAR) AS c FROM places', find)!.sql).toBe(
    'SELECT c FROM places'
  );
  // Numeric to VARCHAR plans a placeholder and reports the column needing a GPU distinct pass.
  const pending = rewriteResidentSql('SELECT CAST(id AS VARCHAR) AS s FROM places', find)!;
  expect(pending.pendingValueColumns).toEqual(['id']);
  expect(pending.derived[0]).toMatchObject({kind: 'valueLookup', name: 's', sourceColumn: 'id'});
  expect(pending.sql).toBe('SELECT s FROM places');
  expect(
    rewriteResidentSql('SELECT CAST(id AS TIMESTAMP) AS t FROM places', find)!.derived[0]
  ).toMatchObject({kind: 'cast', target: 'timestamp'});
  expect(
    rewriteResidentSql('SELECT CAST(id + 1 AS INTEGER) AS n FROM places', find)!.derived[0]
  ).toMatchObject({kind: 'expression', name: 'n', target: 'sint32'});
  expect(() => rewriteResidentSql('SELECT CAST(count AS DOUBLE) AS id FROM places', find)).toThrow(
    /collides/
  );
  // String to number: converts per label on the CPU, refuses labels that would error in SQL.
  const parsed = rewriteResidentSql('SELECT CAST(digits AS DOUBLE) AS d FROM places', find)!;
  const lookup = parsed.derived[0];
  expect(lookup.kind).toBe('lookup');
  const view = new Float32Array((lookup as {words: Uint32Array}).words.buffer);
  expect(Array.from(view)).toEqual([1, 22, 3]);
  expect(() => makeResidentSqlStringCastLookup(['1', 'x'], 'float32')).toThrow(/would fail in SQL/);
  expect(() => makeResidentSqlStringCastLookup(['1.5'], 'sint32')).toThrow(/does not fit/);
});

const FORMATS: ResidentSqlScalarFormat[] = ['float32', 'sint32', 'uint32'];
const TARGETS = ['float32', 'sint32', 'uint32', 'bool'] as const;

function wordsFor(format: ResidentSqlScalarFormat): number[] {
  const view = new DataView(new ArrayBuffer(4));
  const write = (value: number) => {
    if (format === 'float32') view.setFloat32(0, value, true);
    else if (format === 'sint32') view.setInt32(0, value, true);
    else view.setUint32(0, value >>> 0, true);
    return view.getUint32(0, true);
  };
  const values = [
    0, 1, -1, 0.4, 0.5, 1.5, 2.5, -0.5, -0.4, -1.5, 3, 255, 65536, 2147483647, -2147483648,
    4294967295, 16777217, 1e10, -1e10
  ];
  const words = values
    .filter(value => format === 'float32' || Number.isInteger(value))
    .filter(value => format !== 'sint32' || (value >= -2147483648 && value <= 2147483647))
    .filter(value => format !== 'uint32' || (value >= 0 && value <= 4294967295))
    .map(write);
  if (format === 'float32') {
    words.push(0x7fc00000, 0x7f800000, 0xff800000); // NaN, +Infinity, -Infinity
  }
  return words;
}

function wordValue(format: ResidentSqlScalarFormat, word: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setUint32(0, word, true);
  return format === 'float32'
    ? view.getFloat32(0, true)
    : format === 'sint32'
      ? view.getInt32(0, true)
      : word;
}

test('CAST kernel mirror (word level) matches the value-level oracle', () => {
  for (const source of FORMATS) {
    for (const target of TARGETS) {
      for (const word of wordsFor(source)) {
        const expected = castResidentSqlValue(wordValue(source, word), source, target);
        const mirrored = mirrorResidentSqlCastKernel(source, target, word);
        expect(mirrored.valid, `${source}->${target} word ${word}`).toBe(expected.valid);
        if (!expected.valid) {
          expect(mirrored.word).toBe(0);
          continue;
        }
        const resultFormat: ResidentSqlScalarFormat = target === 'bool' ? 'uint32' : target;
        expect(wordValue(resultFormat, mirrored.word), `${source}->${target} word ${word}`).toBe(
          resultFormat === 'float32' ? Math.fround(expected.value) : expected.value
        );
      }
    }
  }
  // Validity from the source is carried through.
  expect(mirrorResidentSqlCastKernel('sint32', 'float32', 5, false).valid).toBe(false);
});

test('CAST rounds half away from zero and refuses out-of-range values as null', () => {
  expect(castResidentSqlValue(2.5, 'float32', 'sint32')).toEqual({value: 3, valid: true});
  expect(castResidentSqlValue(-2.5, 'float32', 'sint32')).toEqual({value: -3, valid: true});
  expect(castResidentSqlValue(-0.4, 'float32', 'uint32').valid).toBe(true);
  expect(castResidentSqlValue(-0.5, 'float32', 'uint32').valid).toBe(false);
  expect(castResidentSqlValue(3e9, 'float32', 'sint32').valid).toBe(false);
  expect(castResidentSqlValue(NaN, 'float32', 'sint32').valid).toBe(false);
  expect(castResidentSqlValue(-1, 'sint32', 'uint32').valid).toBe(false);
  expect(castResidentSqlValue(3000000000, 'uint32', 'sint32').valid).toBe(false);
  expect(castResidentSqlValue(0, 'float32', 'bool').value).toBe(0);
  for (const source of FORMATS)
    for (const target of TARGETS)
      for (const validity of [true, false])
        expect(getResidentSqlCastKernelBody(source, target, validity)).toContain(
          'validityOut[index]'
        );
});

test('GROUP BY adds or reuses a COUNT(*) presence column to drop empty dictionary slots', () => {
  const added = rewriteResidentSql(
    'SELECT city, SUM(amount) AS total FROM places GROUP BY city',
    find
  )!;
  expect(added.groupPresenceColumn).toBe(`${RESIDENT_SQL_HIDDEN_PREFIX}group_count`);
  expect(added.sql).toBe(
    `SELECT city, SUM(amount) AS total, COUNT(*) AS ${added.groupPresenceColumn} FROM places GROUP BY city`
  );
  const reused = rewriteResidentSql('SELECT city, COUNT(*) AS n FROM places GROUP BY city', find);
  expect(reused!.groupPresenceColumn).toBe('n');
  const named = rewriteResidentSql(
    "SELECT city, COUNT(*) AS n FROM places WHERE city LIKE 'B%' GROUP BY city",
    find
  )!;
  expect(named.groupPresenceColumn).toBe('n');
  expect(named.sql).not.toContain('group_count');
  expect(
    rewriteResidentSql('SELECT city, COUNT(*) FROM places GROUP BY city', find)!.groupPresenceColumn
  ).toBe('count_all');
});

test('string expressions over three columns plan one multi-column lookup (bounded product)', () => {
  const wide: ResidentSqlTableInfo = {
    name: 'places',
    columns: {
      a: {format: 'uint32', dictionary: ['x', 'y']},
      b: {format: 'uint32', dictionary: ['1', '2', '3']},
      c: {format: 'uint32', dictionary: ['p', 'q']},
      d: {format: 'uint32', dictionary: ['m', 'n']},
      n: {format: 'float32'}
    }
  };
  const lookup = (name: string) => (name === 'places' ? wide : undefined);
  const three = rewriteResidentSql('SELECT a || b || c AS key FROM places', lookup)!;
  const spec = three.derived.find(entry => entry.kind === 'pairLookup');
  expect(spec).toMatchObject({extraColumns: ['c'], extraSizes: [2], firstSize: 2, secondSize: 3});
  expect(spec && 'words' in spec && spec.words.length).toBe(2 * 3 * 2);
  expect(three.sql).toBe('SELECT key FROM places');
  const grouped = rewriteResidentSql(
    'SELECT UPPER(a || b || c) AS key, COUNT(*) AS rows FROM places GROUP BY UPPER(a || b || c)',
    lookup
  )!;
  expect(grouped.derived.filter(entry => entry.kind === 'pairLookup')).toHaveLength(1);
  const four = rewriteResidentSql('SELECT a || b || c || d AS key FROM places', lookup)!;
  expect(four.derived.find(entry => entry.kind === 'pairLookup')).toMatchObject({
    extraColumns: ['c', 'd']
  });
  expect(() => rewriteResidentSql('SELECT a || n FROM places', lookup)).toThrow(/mix dictionary/);
});
