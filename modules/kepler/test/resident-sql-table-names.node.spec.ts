// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  assignResidentSqlTableNames,
  rewriteQuotedResidentTableNames
} from '../../../examples/deck/kepler-compute/src/resident-sql-gpu-runtime';

describe('resident SQL table names', () => {
  it('gives ids that sanitize alike distinct internal names', () => {
    const names = assignResidentSqlTableNames(['a-b', 'a b', 'a_b']);
    expect(names.get('a-b')).toBe('a_b_2');
    expect(names.get('a b')).toBe('a_b_3');
    expect(names.has('a_b')).toBe(false);
    expect(new Set(names.values()).size).toBe(2);
  });

  it('routes each quoted id to its own table', () => {
    const names = assignResidentSqlTableNames(['a-b', 'a b']);
    expect(rewriteQuotedResidentTableNames('SELECT id FROM "a b"', names)).toBe(
      'SELECT id FROM a_b_2'
    );
    expect(rewriteQuotedResidentTableNames('SELECT id FROM "a-b"', names)).toBe(
      'SELECT id FROM a_b'
    );
  });

  it('keeps output aliases, column names and string literals', () => {
    const names = assignResidentSqlTableNames(['a-b']);
    expect(rewriteQuotedResidentTableNames('SELECT id AS "a-b" FROM "a-b"', names)).toBe(
      'SELECT id AS "a-b" FROM a_b'
    );
    expect(
      rewriteQuotedResidentTableNames('SELECT "a-b" FROM "a-b" WHERE x = \'"a-b"\'', names)
    ).toBe('SELECT "a-b" FROM a_b WHERE x = \'"a-b"\'');
  });

  it('rewrites qualifiers and joins', () => {
    const names = assignResidentSqlTableNames(['a-b', 'c-d']);
    expect(
      rewriteQuotedResidentTableNames(
        'SELECT "a-b".id FROM "a-b" JOIN "c-d" ON "a-b".id = "c-d".id',
        names
      )
    ).toBe('SELECT a_b.id FROM a_b JOIN c_d ON a_b.id = c_d.id');
  });
});
