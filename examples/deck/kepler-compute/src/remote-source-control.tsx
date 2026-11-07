// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useRef, useState} from 'react';
import {loadRemoteSource, type RemoteLoadProgress} from './remote-source-loader';
import {loadResumableRemoteSource} from './remote-resumable-source';
import {parseAutomaticPolicyConfig} from './automatic-recovery-policy';

const AUTOMATIC_POLICY = parseAutomaticPolicyConfig(globalThis.location?.search ?? '');
import type {SourceCandidate} from './source-workspace-panel';

/** "Open from URL" for CSV, GeoJSON, Parquet and Arrow IPC with progress, cancel and errors. */
export function RemoteSourceControl({onSource}: {onSource(source: SourceCandidate): void}) {
  const [url, setUrl] = useState('');
  const [progress, setProgress] = useState<RemoteLoadProgress | null>(null);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const isLoading = progress !== null;
  const start = () => {
    controller.current?.abort();
    const abortController = new AbortController();
    controller.current = abortController;
    setError('');
    setStatus('');
    setProgress({loadedBytes: 0, totalBytes: null});
    void (AUTOMATIC_POLICY.remoteRetry ? loadResumableRemoteSource : loadRemoteSource)(url.trim(), {
      signal: abortController.signal,
      onProgress: next => {
        if (controller.current === abortController) setProgress(next);
      }
    })
      .then(source => {
        if (controller.current !== abortController) return;
        onSource(source);
        setStatus(`Loaded ${source.label}: ${source.table.numRows.toLocaleString()} rows`);
      })
      .catch(failure => {
        if (controller.current !== abortController) return;
        setError(
          abortController.signal.aborted
            ? 'Download cancelled.'
            : failure instanceof Error
              ? failure.message
              : String(failure)
        );
      })
      .finally(() => {
        if (controller.current === abortController) setProgress(null);
      });
  };
  const percent =
    progress?.totalBytes && progress.totalBytes > 0
      ? Math.round((100 * progress.loadedBytes) / progress.totalBytes)
      : null;
  return (
    <fieldset data-testid="remote-source-control" disabled={false}>
      <legend>Open from URL</legend>
      <input
        type="url"
        aria-label="Source URL"
        placeholder="https://… .csv .geojson .parquet .arrow"
        value={url}
        disabled={isLoading}
        onChange={event => setUrl(event.target.value)}
      />
      {isLoading ? (
        <button type="button" onClick={() => controller.current?.abort()}>
          Cancel
        </button>
      ) : (
        <button type="button" disabled={!url.trim()} onClick={start}>
          Open URL
        </button>
      )}
      {isLoading && (
        <progress
          aria-label="Download progress"
          value={percent ?? undefined}
          max={percent === null ? undefined : 100}
        />
      )}
      {isLoading && (
        <span data-testid="remote-source-progress">
          {(progress.loadedBytes / (1024 * 1024)).toFixed(1)} MiB
          {percent === null ? '' : ` (${percent}%)`}
        </span>
      )}
      {status && <p data-testid="remote-source-status">{status}</p>}
      {error && (
        <p role="alert" data-testid="remote-source-error">
          {error}
        </p>
      )}
    </fieldset>
  );
}
