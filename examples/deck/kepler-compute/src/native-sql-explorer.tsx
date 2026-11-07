// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {lazy, Suspense, useEffect, useState} from 'react';
import {DEFAULT_DUCKDB_SQL, type DuckDBRuntime, type DuckDBResultSnapshot} from './duckdb-runtime';
import {runSqlWithFallback, type SqlPlanner, type SqlRoute} from './sql-backend-router';

// Kepler's own Monaco SQL editor, loaded lazily so the editor never enters the initial bundle.
const LazyMonacoEditor = lazy(async () => {
  const components = await import('@kepler.gl/duckdb/components');
  return {default: components.MonacoEditor};
});

/** GPU execution of a planned statement; resolves with a published-dataset-ready summary. */
export type NativeSqlGpuExecutor<Query> = {
  /** Plans without uploading or reading GPU data; null when no resident table is registered. */
  readonly planner: SqlPlanner<Query> | null;
  /** Registered resident tables and datasets left out, for the explorer's table list. */
  describeTables?(): {
    readonly tables: readonly {name: string; datasetId: string; columnCount: number}[];
    readonly skipped: readonly {datasetId: string; reason: string}[];
  };
  /** Runs the planned statement on the GPU, publishes it, and returns the number of result rows. */
  execute(query: Query): Promise<number>;
  /** How the last result was produced, e.g. resident or the readback reason. */
  describeLastResult?(): string;
};

export function describeSqlRoute<Query>(route: SqlRoute<Query>): string {
  return route.backend === 'gpu'
    ? 'GPU planner accepted the statement; data stays resident.'
    : `DuckDB fallback: ${route.reason}. Source and result cross as explicit Arrow transfers.`;
}

/**
 * Native Kepler SQL editor surface over the resident SQL router: GPU planner first, DuckDB-WASM
 * fallback with the stated reason, results published through the existing native publication path.
 */
export function NativeSqlExplorer<Query>({
  runtime,
  gpu,
  onPrepareResult
}: {
  runtime: DuckDBRuntime;
  gpu?: NativeSqlGpuExecutor<Query>;
  onPrepareResult?: (result: DuckDBResultSnapshot) => void | Promise<void>;
}) {
  const [, refresh] = useState(0);
  const [sql, setSql] = useState(DEFAULT_DUCKDB_SQL);
  const [route, setRoute] = useState<SqlRoute<Query> | null>(null);
  const [gpuRowCount, setGpuRowCount] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [running, setRunning] = useState(false);
  useEffect(() => runtime.subscribe(() => refresh(value => value + 1)), [runtime]);
  const busy = running || runtime.status === 'busy' || runtime.status === 'starting';

  const runStatement = async () => {
    setRunning(true);
    setError('');
    setGpuRowCount(null);
    try {
      const outcome = await runSqlWithFallback({
        sql,
        planner: gpu?.planner ?? null,
        executeGpu: query => (gpu as NativeSqlGpuExecutor<Query>).execute(query),
        onRoute: setRoute,
        executeDuckDb: async () => {
          if (runtime.status === 'disabled') await runtime.initialize();
          if (!runtime.sourceRegistered) await runtime.registerSource();
          await runtime.runSql(sql);
        }
      });
      if (outcome.backend === 'gpu') setGpuRowCount(outcome.rowCount);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setRunning(false);
    }
  };
  const publishResult = async () => {
    try {
      setError('');
      const result = runtime.currentResult;
      if (!result) throw new Error('Run SQL before publishing its result');
      // importResult uploads once; the runtime subscription in main.tsx then attaches the dataset.
      runtime.importResult(result.generation);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };
  const resultInfo = runtime.resultInfo;
  return (
    <section aria-label="Native SQL Data Explorer" data-native-sql-explorer="true">
      <h2 style={{fontSize: 16}}>SQL Data Explorer</h2>
      <div style={{height: 220, background: 'rgb(30, 30, 30)'}}>
        <Suspense fallback={<p>Loading editor</p>}>
          <LazyMonacoEditor
            code={sql}
            onChange={value => setSql(value ?? '')}
            onRunQuery={() => void runStatement()}
          />
        </Suspense>
      </div>
      <button disabled={busy} onClick={() => void runStatement()}>
        Run SQL
      </button>{' '}
      <button disabled={busy || !resultInfo} onClick={() => void publishResult()}>
        Publish result
      </button>
      {onPrepareResult && (
        <>
          {' '}
          <button
            disabled={busy || !runtime.currentResult}
            onClick={() => {
              const result = runtime.currentResult;
              if (result)
                void Promise.resolve(onPrepareResult(result)).catch(f => setError(String(f)));
            }}
          >
            Prepare result for workspace
          </button>
        </>
      )}
      {gpu?.describeTables && (
        <p data-sql-tables="true">
          GPU tables:{' '}
          {gpu
            .describeTables()
            .tables.map(table => `"${table.name}" (${table.columnCount} columns)`)
            .join(', ') || 'none'}
          {gpu.describeTables().skipped.length
            ? `. Not queryable on GPU: ${gpu
                .describeTables()
                .skipped.map(entry => `${entry.datasetId} (${entry.reason})`)
                .join(', ')}`
            : ''}
        </p>
      )}
      {route && (
        <p role="status" data-sql-backend={route.backend}>
          {describeSqlRoute(route)}
        </p>
      )}
      <p role="status">
        {runtime.status}
        {gpuRowCount !== null
          ? ` · ${gpuRowCount} GPU result rows${gpu?.describeLastResult ? ` (${gpu.describeLastResult()})` : ''}`
          : ''}
        {resultInfo ? ` · ${resultInfo.rowCount} DuckDB result rows` : ''}
      </p>
      {resultInfo && (
        <pre style={{overflow: 'auto', maxHeight: 180}}>
          {JSON.stringify(
            runtime.previewRows(),
            (_, value) => (typeof value === 'bigint' ? value.toString() : value),
            2
          )}
        </pre>
      )}
      {runtime.importedDataset && (
        <p>
          Published: {runtime.importedDataset.id}, revision {runtime.importedDataset.revision},{' '}
          {runtime.importedDataset.rowCount} rows.
        </p>
      )}
      {(error || runtime.error) && <p role="alert">{error || runtime.error}</p>}
    </section>
  );
}
