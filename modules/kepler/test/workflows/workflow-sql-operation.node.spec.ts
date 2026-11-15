// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  getWorkflowSqlOperations,
  getWorkflowSqlProblem,
  getWorkflowSqlTableNames,
  WORKFLOW_SQL_OPERATION_ID
} from '../../src/workflows/workflow-sql-operation';
import {
  WorkflowError,
  type WorkflowHost,
  type WorkflowRunContext,
  type WorkflowSqlRequest
} from '../../src/workflows/workflow-types';
import {FIXTURE_KINDS, makeFixtureHost, makeFixturePoints} from './workflow-fixtures';

const [operation] = getWorkflowSqlOperations();

function makeContext(host: WorkflowHost = makeFixtureHost()): WorkflowRunContext {
  return {
    device: null,
    engine: 'auto',
    host,
    signal: new AbortController().signal,
    workflowId: 'workflow',
    stepId: 'sql'
  };
}

describe('workflow SQL guard', () => {
  test('accepts SELECT, CTEs and syntax-like text inside strings, identifiers and comments', () => {
    for (const query of [
      'SELECT * FROM input1',
      ' /* preface */ with points AS (SELECT * FROM input1) SELECT * FROM points;',
      "SELECT 'DROP; read_csv(''file'') -- /*' FROM input1",
      'SELECT "DELETE", "read_parquet", "a"";DROP" FROM input1',
      'SELECT * FROM input1 -- ; DELETE FROM input1\n',
      'SELECT /* DROP; /* read_csv( */ INSERT */ * FROM input1; /* LOAD */',
      'SELECT updated, create_time, read_count, scanner FROM input1',
      "SELECT * FROM input1 WHERE name = 'file.csv'",
      'SELECT glob_id FROM input1',
      "SELECT 'getenv(' FROM input1",
      "SELECT concat(name, 'file.csv'), 'other.csv' FROM input1",
      "SELECT * FROM input1 WHERE name IN ('file.csv', 'other.csv')",
      "SELECT * FROM input1 ORDER BY name, 'file.csv'",
      "SELECT extract(YEAR FROM '2026-01-01'::DATE) FROM input1",
      "SELECT * FROM (SELECT 'file.csv' AS name FROM input1) AS names, input2"
    ]) {
      expect(getWorkflowSqlProblem(query), query).toBeNull();
    }
  });

  test('refuses second statements even after a comment and requires SELECT', () => {
    for (const query of [
      '',
      '-- SELECT 1',
      'VALUES (1)',
      'WITH points AS (VALUES (1)) TABLE points',
      'SELECT 1;;',
      'SELECT 1; SELECT 2',
      "SELECT 1; 'second statement'",
      'SELECT 1; "second statement"',
      'SELECT 1; -- harmless comment\nSELECT 2',
      'SELECT 1; /* harmless comment */ DROP TABLE input1'
    ]) {
      expect(getWorkflowSqlProblem(query), query).toBe('must be a single SELECT statement');
    }
  });

  test('refuses each modification, configuration and external statement keyword', () => {
    for (const keyword of [
      'INSERT',
      'UPDATE',
      'DELETE',
      'DROP',
      'CREATE',
      'ALTER',
      'TRUNCATE',
      'MERGE',
      'GRANT',
      'REVOKE',
      'CALL',
      'SET',
      'RESET',
      'VACUUM',
      'CHECKPOINT',
      'DETACH',
      'USE',
      'COPY',
      'ATTACH',
      'INSTALL',
      'LOAD',
      'PRAGMA',
      'EXPORT',
      'IMPORT'
    ]) {
      expect(getWorkflowSqlProblem(`WITH data AS (${keyword.toLowerCase()} input1) SELECT 1`)).toBe(
        `may not use ${keyword}`
      );
    }
  });

  test('refuses external readers, host access and indirect queries', () => {
    for (const reader of [
      'read_csv',
      'read_parquet',
      'read_json_auto',
      'sqlite_scan',
      'httpfs',
      'glob',
      'getenv',
      'read_text',
      'read_blob',
      'sniff_csv',
      'query',
      'query_table',
      'current_setting'
    ]) {
      expect(
        getWorkflowSqlProblem(`SELECT * FROM ${reader.toUpperCase()} /* comment */ ('url')`)
      ).toBe(`may not read files or URLs (${reader})`);
    }
    expect(getWorkflowSqlProblem("SELECT * FROM glob('*')")).toBe(
      'may not read files or URLs (glob)'
    );
    expect(getWorkflowSqlProblem("SELECT getenv('HOME')")).toBe(
      'may not read files or URLs (getenv)'
    );
  });

  test('refuses quoted file names in FROM lists, joins and nested queries', () => {
    const problem = 'may not read files or URLs (a quoted file name after FROM)';
    for (const query of [
      "SELECT * FROM 'file.csv'",
      "SELECT * FROM /* comment */ 'file.csv'",
      "SELECT * FROM input1, 'file.csv'",
      "SELECT * FROM input1, /* comment */ 'https://example.com/file.csv'",
      'SELECT * FROM "input1", \'file.csv\'',
      "SELECT * FROM input1 JOIN 'file.csv' ON true",
      "SELECT * FROM input1 LEFT OUTER JOIN 'file.csv' ON true",
      "SELECT * FROM input1 RIGHT JOIN 'file.csv' ON true",
      "SELECT * FROM input1 FULL JOIN 'file.csv' ON true",
      "SELECT * FROM input1 CROSS JOIN 'file.csv'",
      "SELECT * FROM input1 NATURAL JOIN 'file.csv'",
      "SELECT * FROM input1 POSITIONAL JOIN 'file.csv'",
      "SELECT * FROM input1, LATERAL 'file.csv'",
      "SELECT * FROM (input1, 'file.csv')",
      "SELECT * FROM (SELECT * FROM 'file.csv') AS files",
      "SELECT * FROM input1 WHERE EXISTS (SELECT * FROM 'file.csv')",
      "WITH points AS (SELECT * FROM input1) SELECT * FROM points, 'file.csv'"
    ]) {
      expect(getWorkflowSqlProblem(query), query).toBe(problem);
    }
    expect(() => operation.normalizeParameters({query: "SELECT * FROM 'file.csv'"})).toThrow(
      expect.objectContaining({
        code: 'parameter-out-of-range',
        message: `SQL: Query ${problem}.`,
        details: {parameter: 'query'}
      })
    );
  });

  test('refuses unterminated quoted text and comments and enforces 4000 characters', () => {
    expect(getWorkflowSqlProblem("SELECT 'open''string")).toBe('must close every string literal');
    expect(getWorkflowSqlProblem('SELECT "open')).toBe('must close every quoted identifier');
    expect(getWorkflowSqlProblem('SELECT 1 /* open')).toBe('must close every block comment');
    expect(getWorkflowSqlProblem('SELECT 1'.padEnd(4000))).toBeNull();
    expect(getWorkflowSqlProblem('SELECT 1'.padEnd(4001))).toBe(
      'must be at most 4000 characters; got 4001'
    );
  });

  test('detects case-insensitive whole table names once, in first-use order', () => {
    expect(
      getWorkflowSqlTableNames(
        `SELECT 'input1; ''input2', "value" FROM INPUT3 /* input2 */
       JOIN "input1" ON true JOIN input3 ON true -- input2
       WHERE input10 = 1 AND my_input2 = 2 AND input2_suffix = 3`
      )
    ).toEqual(['input3', 'input1']);
    expect(getWorkflowSqlTableNames('SELECT * FROM input2, input1, INPUT2, input3')).toEqual([
      'input2',
      'input1',
      'input3'
    ]);
    expect(getWorkflowSqlTableNames("SELECT 'input1'")).toEqual([]);
  });
});

describe('workflow SQL operation', () => {
  test('declares bounded ports, host execution and validated defaults', () => {
    expect(operation).toMatchObject({
      id: WORKFLOW_SQL_OPERATION_ID,
      title: 'SQL',
      category: 'table',
      origin: 'builtin',
      engine: 'host',
      version: '1',
      outputs: [{name: 'result', kind: 'features', maximumRows: 100000}]
    });
    expect(operation.inputs).toEqual(
      ['input1', 'input2', 'input3'].map((name, index) => ({
        name,
        label: name,
        accepts: FIXTURE_KINDS,
        maximumRows: 100000,
        maximumVertices: 2000000,
        ...(index > 0 ? {optional: true} : {})
      }))
    );
    expect(operation.normalizeParameters({})).toEqual({
      query: 'SELECT * FROM input1',
      maximumRows: 100000
    });
    expect(operation.normalizeParameters({maximumRows: '1'})['maximumRows']).toBe(1);
    for (const maximumRows of [0, 100001, 1.5]) {
      expect(() => operation.normalizeParameters({maximumRows})).toThrow(
        expect.objectContaining({
          code: 'parameter-out-of-range',
          details: {parameter: 'maximumRows'}
        })
      );
    }
    expect(() => operation.normalizeParameters({query: "SELECT * FROM read_csv('file')"})).toThrow(
      expect.objectContaining({
        code: 'parameter-out-of-range',
        message: 'SQL: Query may not read files or URLs (read_csv).',
        details: {parameter: 'query'}
      })
    );
    expect(() => operation.normalizeParameters({query: 'SELECT 1'.padEnd(4001)})).toThrow(
      expect.objectContaining({
        code: 'parameter-out-of-range',
        message: 'SQL: Query must be at most 4000 characters; got 4001.',
        details: {parameter: 'query'}
      })
    );
  });

  test('passes only named connected inputs and the signal, preserving host data', async () => {
    const host = makeFixtureHost();
    const calls: Array<{request: WorkflowSqlRequest; signal: AbortSignal}> = [];
    const data = makeFixturePoints(2);
    host.runSql = async (request, signal) => {
      calls.push({request, signal});
      return data;
    };
    const context = makeContext(host);
    const query = 'SELECT * FROM input3 JOIN input1 USING (point_id)';
    const inputs = {input1: data, input2: makeFixturePoints(1), input3: data};
    const result = await operation.run(
      context,
      inputs,
      operation.normalizeParameters({query, maximumRows: 2})
    );
    expect(calls).toEqual([
      {
        request: {query, tables: {input3: data, input1: data}, maximumRows: 2},
        signal: context.signal
      }
    ]);
    expect(Object.keys(calls[0].request.tables)).toEqual(['input3', 'input1']);
    expect(calls[0].request.tables['input1']).toBe(data);
    expect(result).toEqual({
      outputs: {result: data},
      diagnostics: [],
      summary: {rows: 2, columns: 3},
      engine: 'host',
      dispatchedContributors: []
    });
    expect(result.outputs['result']).toBe(data);
  });

  test('reports a missing SQL engine or named connection before calling the host', async () => {
    await expect(
      operation.run(makeContext(), {}, operation.normalizeParameters({}))
    ).rejects.toMatchObject({
      code: 'host-capability-missing',
      message: 'SQL: this application has no SQL engine.'
    });
    const host = makeFixtureHost();
    let calls = 0;
    host.runSql = async () => {
      calls++;
      return makeFixturePoints(1);
    };
    await expect(
      operation.run(
        makeContext(host),
        {input1: makeFixturePoints(1)},
        operation.normalizeParameters({query: 'SELECT * FROM input2'})
      )
    ).rejects.toMatchObject({
      code: 'input-required',
      message: 'SQL: input2 must be connected.',
      details: {port: 'input2'}
    });
    expect(calls).toBe(0);
  });

  test('enforces the requested output row capacity', async () => {
    const host = makeFixtureHost();
    host.runSql = async () => makeFixturePoints(2);
    await expect(
      operation.run(
        makeContext(host),
        {input1: makeFixturePoints(1)},
        operation.normalizeParameters({maximumRows: 1})
      )
    ).rejects.toMatchObject({
      code: 'capacity-exceeded',
      message: 'SQL: Result must have at most 1 rows (Maximum rows); got 2.',
      details: {port: 'result'}
    });
  });

  test('wraps host failures with their cause and preserves WorkflowError codes', async () => {
    const host = makeFixtureHost();
    const cause = new Error('invalid column');
    host.runSql = async () => {
      throw cause;
    };
    const inputs = {input1: makeFixturePoints(1)};
    await expect(
      operation.run(makeContext(host), inputs, operation.normalizeParameters({}))
    ).rejects.toMatchObject({
      code: 'operation-failed',
      message: 'SQL: invalid column',
      cause,
      details: {cause}
    });
    const workflowError = new WorkflowError('cancelled', 'SQL: cancelled.');
    host.runSql = async () => {
      throw workflowError;
    };
    await expect(
      operation.run(makeContext(host), inputs, operation.normalizeParameters({}))
    ).rejects.toBe(workflowError);
    const unprintableCause: unknown = Object.create(null);
    host.runSql = async () => {
      throw unprintableCause;
    };
    await expect(
      operation.run(makeContext(host), inputs, operation.normalizeParameters({}))
    ).rejects.toMatchObject({
      code: 'operation-failed',
      message: 'SQL: Unknown error',
      cause: unprintableCause,
      details: {cause: unprintableCause}
    });
  });
});
