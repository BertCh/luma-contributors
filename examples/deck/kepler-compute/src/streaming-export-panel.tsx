// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {wrapPageSourceWithColumnEdits} from './resident-export-column-edits';
import {residentColumnEditStore} from './resident-column-edits';
import {useRef, useState} from 'react';
import type {ComputeRuntime} from './compute-runtime';
import {getSandboxExplorer} from './sandbox-state';
import {trackOperation} from './operation-tracker';
import {getNativeOnlyDatasets} from './native-source-access';
import {exportNativeDataset, type NativeExportSelection} from './native-source-export';
import {
  makeNativePageSource,
  makeResidentPageSource,
  openStreamingSink,
  runStreamingExport,
  STREAMING_EXPORT_DEFAULTS,
  type StreamingExportFormat,
  type StreamingExportProgress
} from './streaming-export';

const NATIVE_PREFIX = 'native:';
const NON_STREAMING_LIMITS = {maximumRows: 1_000_000, maximumOutputBytes: 64 * 1024 * 1024};

/** Native-only tables and unbounded streaming downloads, beside the bounded export controls. */
export function StreamingExportPanel({
  runtime,
  disabled
}: {
  runtime: ComputeRuntime;
  disabled: boolean;
}) {
  const nativeDatasets = getNativeOnlyDatasets();
  const resident = [
    {id: runtime.explorer.dataset.id, label: 'Observations fixture'},
    ...Array.from(runtime.publications.values(), published => ({
      id: published.dataset.id,
      label: published.label
    }))
  ];
  const [choice, setChoice] = useState(resident[0].id);
  const [format, setFormat] = useState<StreamingExportFormat>('csv');
  const [selection, setSelection] = useState<NativeExportSelection>('filtered');
  const [progress, setProgress] = useState<StreamingExportProgress | null>(null);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const cancellation = useRef<AbortController | null>(null);
  const isNative = choice.startsWith(NATIVE_PREFIX);
  const nativeDataset = isNative
    ? nativeDatasets.find(dataset => dataset.id === choice.slice(NATIVE_PREFIX.length))
    : undefined;
  const isAvailable = isNative ? Boolean(nativeDataset) : resident.some(item => item.id === choice);
  const filenameBase = `kepler-${isNative ? choice.slice(NATIVE_PREFIX.length) : choice}`;
  const extension = format === 'arrow' ? 'arrows' : 'csv';

  const run = async (operation: (controller: AbortController) => Promise<void>) => {
    const controller = new AbortController();
    cancellation.current = controller;
    setBusy(true);
    setStatus('');
    setProgress(null);
    try {
      await operation(controller);
    } catch (error) {
      setStatus(
        controller.signal.aborted
          ? 'Export cancelled; no file was published.'
          : error instanceof Error
            ? error.message
            : String(error)
      );
    } finally {
      cancellation.current = null;
      setBusy(false);
    }
  };
  const streamDownload = () =>
    run(async controller => {
      // The picker needs the click's user activation, so it opens before any readback.
      const sink = await openStreamingSink(`${filenameBase}.${extension}`, format);
      const source = nativeDataset
        ? makeNativePageSource(nativeDataset, selection)
        : wrapPageSourceWithColumnEdits(
            makeResidentPageSource(runtime.session, getSandboxExplorer(runtime, choice)),
            () => residentColumnEditStore.get(choice)
          );
      const tracked = trackOperation(`Stream download ${filenameBase}`, () => controller.abort());
      const result = await runStreamingExport({
        source,
        format,
        sink,
        signal: controller.signal,
        onProgress: value => {
          setProgress(value);
          tracked.update({
            fraction: value.totalRows ? value.rowsWritten / value.totalRows : undefined,
            detail: `${value.rowsWritten.toLocaleString()} of ${value.totalRows.toLocaleString()} rows`
          });
        }
      }).finally(() => tracked.finish());
      setStatus(
        `${result.rowsWritten.toLocaleString()} rows, ${result.bytesWritten.toLocaleString()} bytes streamed in ${result.pagesWritten} pages via ${sink.kind === 'file-system' ? 'the file system' : 'an in-memory download (bounded)'}.`
      );
    });
  const exportNative = () =>
    run(async controller => {
      const result = await exportNativeDataset(nativeDataset!, {
        format,
        selection,
        ...NON_STREAMING_LIMITS,
        signal: controller.signal
      });
      const url = URL.createObjectURL(new Blob([result.bytes]));
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `${filenameBase}-${result.datasetRevision}.${extension}`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
      setStatus(`${result.rowCount.toLocaleString()} native rows exported (${format}).`);
    });

  return (
    <section aria-label="Stream download" data-testid="streaming-export">
      <h3>Native tables and stream download</h3>
      <p className="hint">
        Native tables that never reached the GPU export their original values (nulls kept, native
        field order; filtered rows follow the Kepler filter). Stream download writes pages through a
        file handle when the browser supports one and has no row or output limit; otherwise it falls
        back to a download held in memory up to{' '}
        {Math.round(STREAMING_EXPORT_DEFAULTS.maximumBlobBytes / (1024 * 1024))} MiB.
      </p>
      <label>
        Stream dataset
        <select
          aria-label="Stream dataset"
          value={choice}
          disabled={busy || disabled}
          onChange={event => setChoice(event.target.value)}
        >
          {resident.map(item => (
            <option key={item.id} value={item.id}>
              {item.label} (GPU)
            </option>
          ))}
          {nativeDatasets.map(dataset => (
            <option key={dataset.id} value={`${NATIVE_PREFIX}${dataset.id}`}>
              {dataset.label} (native table)
            </option>
          ))}
        </select>
      </label>
      <label>
        Stream format
        <select
          aria-label="Stream format"
          value={format}
          disabled={busy || disabled}
          onChange={event => setFormat(event.target.value === 'arrow' ? 'arrow' : 'csv')}
        >
          <option value="csv">CSV</option>
          <option value="arrow">Arrow IPC stream</option>
        </select>
      </label>
      {isNative ? (
        <label>
          Native rows
          <select
            aria-label="Native rows"
            value={selection}
            disabled={busy || disabled}
            onChange={event => setSelection(event.target.value === 'all' ? 'all' : 'filtered')}
          >
            <option value="filtered">Filtered rows</option>
            <option value="all">All rows</option>
          </select>
        </label>
      ) : null}
      <div className="actions">
        {isNative ? (
          <button disabled={busy || disabled || !isAvailable} onClick={() => void exportNative()}>
            Export native table
          </button>
        ) : null}
        <button disabled={busy || disabled || !isAvailable} onClick={() => void streamDownload()}>
          Stream download
        </button>
        {busy ? <button onClick={() => cancellation.current?.abort()}>Cancel stream</button> : null}
      </div>
      {progress ? (
        <progress
          aria-label="Stream progress"
          max={Math.max(1, progress.totalRows)}
          value={progress.rowsWritten}
        />
      ) : null}
      {progress ? (
        <p className="hint" data-testid="streaming-progress">
          {progress.rowsWritten.toLocaleString()} of {progress.totalRows.toLocaleString()} rows,{' '}
          {progress.bytesWritten.toLocaleString()} bytes
        </p>
      ) : null}
      {status ? (
        <p role="status" className="hint">
          {status}
        </p>
      ) : null}
    </section>
  );
}
