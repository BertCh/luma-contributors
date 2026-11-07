// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Structural view of `GPUSQLContext`; parsing and planning never upload or read GPU data. */
export type SqlPlanner<Query> = {query(sql: string): Query};

export type SqlRoute<Query> =
  | {readonly backend: 'gpu'; readonly query: Query}
  | {
      readonly backend: 'duckdb';
      /** Why GPU planning declined; shown to the user beside the explicit Arrow transfer notice. */
      readonly reason: string;
      /** True when the CPU backend implies an Arrow export of the registered source and result. */
      readonly requiresArrowTransfer: true;
    };

/**
 * Routes one SQL statement. Statements the GPU planner accepts stay resident; all others run on the
 * chosen CPU backend and the caller must surface the explicit Arrow transfers it implies.
 */
export function planSqlRoute<Query>(
  sql: string,
  planner: SqlPlanner<Query> | null
): SqlRoute<Query> {
  if (!planner) {
    return {
      backend: 'duckdb',
      reason: 'No resident SQL tables are registered',
      requiresArrowTransfer: true
    };
  }
  try {
    return {backend: 'gpu', query: planner.query(sql)};
  } catch (error) {
    return {
      backend: 'duckdb',
      reason: error instanceof Error ? error.message : String(error),
      requiresArrowTransfer: true
    };
  }
}

/**
 * Thrown by GPU execution when data discovered at run time (not the statement text) makes the GPU
 * plan unusable, for example a dictionary that would exceed its bound. The caller retries the same
 * statement on DuckDB with `reason`; unrelated failures are not refusals and still surface.
 */
export class SqlExecutionRefusal extends Error {
  constructor(readonly reason: string) {
    super(reason);
    this.name = 'SqlExecutionRefusal';
  }
}

export type SqlRunOutcome =
  | {readonly backend: 'gpu'; readonly rowCount: number}
  | {readonly backend: 'duckdb'; readonly reason: string; readonly requiresArrowTransfer: true};

/**
 * Plans, executes on the GPU and, when planning or execution refuses, runs on DuckDB. Execution
 * refusals (SqlExecutionRefusal) retry on DuckDB; any other error propagates.
 */
export async function runSqlWithFallback<Query>(props: {
  sql: string;
  planner: SqlPlanner<Query> | null;
  executeGpu: (query: Query) => Promise<number>;
  executeDuckDb: () => Promise<void>;
  onRoute?: (route: SqlRoute<Query>) => void;
}): Promise<SqlRunOutcome> {
  const route = planSqlRoute(props.sql, props.planner);
  props.onRoute?.(route);
  if (route.backend === 'gpu') {
    try {
      return {backend: 'gpu', rowCount: await props.executeGpu(route.query)};
    } catch (error) {
      if (!(error instanceof SqlExecutionRefusal)) throw error;
      const fallback = {
        backend: 'duckdb',
        reason: error.reason,
        requiresArrowTransfer: true
      } as const;
      props.onRoute?.(fallback);
      await props.executeDuckDb();
      return fallback;
    }
  }
  await props.executeDuckDb();
  return {backend: 'duckdb', reason: route.reason, requiresArrowTransfer: true};
}
