// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {inlineResidentSqlCtes} from '../../../examples/deck/kepler-compute/src/resident-sql-plan';

test('inlines a simple filtered CTE into the main SELECT', () => {
  expect(
    inlineResidentSqlCtes('WITH t AS (SELECT * FROM trips WHERE fare > 5) SELECT a, b FROM t')
  ).toBe('SELECT a, b FROM trips WHERE fare > 5');
});

test('combines CTE and outer predicates and keeps trailing clauses', () => {
  expect(
    inlineResidentSqlCtes(
      'WITH t AS (SELECT a, b FROM trips WHERE a > 1) SELECT a FROM t WHERE b < 3 ORDER BY a LIMIT 5'
    )
  ).toBe('SELECT a FROM trips WHERE (a > 1) AND (b < 3) ORDER BY a LIMIT 5');
});

test('chains CTEs', () => {
  expect(
    inlineResidentSqlCtes(
      'WITH x AS (SELECT * FROM trips WHERE a > 1), y AS (SELECT * FROM x WHERE b > 2) SELECT a FROM y'
    )
  ).toBe('SELECT a FROM trips WHERE (a > 1) AND (b > 2)');
});

test('leaves unsupported shapes unchanged', () => {
  for (const sql of [
    'WITH RECURSIVE t AS (SELECT * FROM trips) SELECT a FROM t',
    'WITH t AS (SELECT a + 1 AS c FROM trips) SELECT c FROM t',
    'WITH t AS (SELECT a FROM trips GROUP BY a) SELECT a FROM t',
    'WITH t AS (SELECT a FROM trips) SELECT t.a FROM t',
    'WITH t AS (SELECT a FROM trips) SELECT * FROM t',
    'SELECT a FROM trips'
  ])
    expect(inlineResidentSqlCtes(sql)).toBe(sql);
});
