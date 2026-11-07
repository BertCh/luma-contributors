// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  planSqlRoute,
  runSqlWithFallback,
  SqlExecutionRefusal
} from '../../../examples/deck/kepler-compute/src/sql-backend-router';

test('SQL routing keeps supported statements on the GPU and explains fallbacks', () => {
  const planner = {
    query(sql: string) {
      if (sql.includes('WINDOW')) throw new Error('window functions are unsupported');
      return {sql};
    }
  };
  expect(planSqlRoute('SELECT 1', planner)).toEqual({backend: 'gpu', query: {sql: 'SELECT 1'}});
  const fallback = planSqlRoute('SELECT WINDOW', planner);
  expect(fallback).toMatchObject({backend: 'duckdb', requiresArrowTransfer: true});
  expect(planSqlRoute('SELECT 1', null).backend).toBe('duckdb');
});

test('an execution-time refusal retries the same statement on DuckDB with the reason', async () => {
  const planner = {query: (sql: string) => ({sql})};
  const routes: string[] = [];
  let duckDbRuns = 0;
  const outcome = await runSqlWithFallback({
    sql: 'SELECT CAST(x AS VARCHAR) FROM t',
    planner,
    executeGpu: async () => {
      throw new SqlExecutionRefusal(
        'CAST to VARCHAR needs 2000000 distinct values (GPU limit 1048576)'
      );
    },
    executeDuckDb: async () => {
      duckDbRuns++;
    },
    onRoute: route => routes.push(route.backend)
  });
  expect(outcome).toMatchObject({backend: 'duckdb', requiresArrowTransfer: true});
  expect(outcome.backend === 'duckdb' && outcome.reason).toMatch(/distinct values/);
  expect(duckDbRuns).toBe(1);
  expect(routes).toEqual(['gpu', 'duckdb']);
});

test('other GPU failures are errors, not fallbacks, and planning refusals skip the GPU', async () => {
  const planner = {query: (sql: string) => ({sql})};
  let duckDbRuns = 0;
  await expect(
    runSqlWithFallback({
      sql: 'SELECT 1',
      planner,
      executeGpu: async () => {
        throw new Error('SQL sources changed during execution');
      },
      executeDuckDb: async () => {
        duckDbRuns++;
      }
    })
  ).rejects.toThrow(/sources changed/);
  expect(duckDbRuns).toBe(0);
  const ran = await runSqlWithFallback({
    sql: 'SELECT 1',
    planner: {
      query: () => {
        throw new Error('window functions are unsupported');
      }
    },
    executeGpu: async () => 1,
    executeDuckDb: async () => {
      duckDbRuns++;
    }
  });
  expect(ran).toMatchObject({backend: 'duckdb', reason: 'window functions are unsupported'});
  expect(duckDbRuns).toBe(1);
  const gpu = await runSqlWithFallback({
    sql: 'SELECT 1',
    planner,
    executeGpu: async () => 7,
    executeDuckDb: async () => {
      duckDbRuns++;
    }
  });
  expect(gpu).toEqual({backend: 'gpu', rowCount: 7});
});
