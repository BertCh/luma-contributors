// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {DEFAULT_DUCKDB_SQL, type DuckDBRuntime, type DuckDBResultSnapshot} from './duckdb-runtime';

export function DuckDBPanel({
  runtime,
  onPrepareResult
}: {
  runtime: DuckDBRuntime;
  /** Revalidate runtime.isResultCurrent(result) before publishing any asynchronous preparation. */
  onPrepareResult?: (result: DuckDBResultSnapshot) => void | Promise<void>;
}) {
  const [, refresh] = useState(0);
  const [sql, setSql] = useState(DEFAULT_DUCKDB_SQL);
  const [error, setError] = useState('');
  const [preparing, setPreparing] = useState(false);
  useEffect(() => runtime.subscribe(() => refresh(value => value + 1)), [runtime]);
  const busy =
    preparing ||
    runtime.status === 'busy' ||
    runtime.status === 'starting' ||
    runtime.status === 'closed';
  const invoke = async (action: () => unknown) => {
    try {
      await action();
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };
  const statistics = runtime.statistics;
  const prepareResult = async () => {
    const result = runtime.currentResult;
    if (!result || !onPrepareResult) throw new Error('Run SQL before preparing its result');
    setPreparing(true);
    try {
      await onPrepareResult(result);
      if (!runtime.isResultCurrent(result))
        throw new Error('SQL result changed while preparing its preview');
    } finally {
      setPreparing(false);
    }
  };
  return (
    <section
      aria-label="DuckDB SQL complement"
      style={{borderTop: '1px solid #405168', marginTop: 18, paddingTop: 10}}
    >
      <h2 style={{fontSize: 16}}>CPU SQL complement</h2>
      <p>
        Optional DuckDB-WASM queries retained source Arrow. Display filters and derived GPU colors
        are separate.
      </p>
      <button
        disabled={runtime.status !== 'disabled'}
        onClick={() => void invoke(() => runtime.initialize())}
      >
        Enable DuckDB
      </button>{' '}
      <button
        disabled={busy || runtime.status === 'disabled' || runtime.sourceRegistered}
        onClick={() => void invoke(() => runtime.registerSource())}
      >
        Register source Arrow
      </button>
      <label htmlFor="duckdb-sql">
        SQL on observations (source revision {runtime.sourceRevision})
      </label>
      <textarea
        id="duckdb-sql"
        rows={5}
        value={sql}
        onChange={event => setSql(event.target.value)}
        style={{width: '100%', boxSizing: 'border-box'}}
      />
      <button
        disabled={busy || !runtime.sourceRegistered}
        onClick={() => void invoke(() => runtime.runSql(sql))}
      >
        Run CPU SQL
      </button>{' '}
      <button
        disabled={busy || !runtime.resultInfo}
        onClick={() => void invoke(() => runtime.importResult())}
      >
        Upload result to GPU
      </button>
      {onPrepareResult && (
        <>
          {' '}
          <button
            disabled={busy || !runtime.currentResult}
            onClick={() => void invoke(prepareResult)}
          >
            Prepare result for workspace
          </button>
        </>
      )}
      <p role="status">
        {runtime.status}
        {runtime.resultInfo ? ` · ${runtime.resultInfo.rowCount} result rows` : ''}
      </p>
      {runtime.resultInfo && (
        <pre style={{overflow: 'auto', maxHeight: 180}}>
          {JSON.stringify(
            runtime.previewRows(),
            (_, value) => (typeof value === 'bigint' ? value.toString() : value),
            2
          )}
        </pre>
      )}
      <p>
        Arrow IPC sizes: source {statistics.sourceArrowBytes} B · results{' '}
        {statistics.resultArrowBytes} B. Explicit GPU uploads: {statistics.uploads} /{' '}
        {statistics.uploadedBytes} B.
      </p>
      <p>
        DuckDB worker/WASM memory is outside the resident GPU budget. Result Arrow buffers are
        accounted until uploaded or replaced.
      </p>
      {runtime.importedDataset && (
        <p>
          Resident: {runtime.importedDataset.id}, revision {runtime.importedDataset.revision},{' '}
          {runtime.importedDataset.rowCount} rows.
        </p>
      )}
      {(error || runtime.error) && <p role="alert">{error || runtime.error}</p>}
    </section>
  );
}
