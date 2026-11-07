// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useRef, useState} from 'react';
import {SandboxSharePanel} from './sandbox-share-panel';
import {SandboxBundlePanel} from './sandbox-bundle-panel';
import {MediaExportPanel} from './media-export-panel';
import {ShareLinkSourcesPanel} from './share-link-sources-panel';
import {StreamingExportPanel} from './streaming-export-panel';
import {exportSelectedSourceWithColumnEdits} from './resident-export-column-edits';
import {residentColumnEditStore} from './resident-column-edits';
import {exportSelectedAnalysis} from './analysis-export';
import {
  exportSandboxSelection,
  getSandboxExplorer,
  replaySandbox,
  saveSandbox,
  serializeSandboxExport,
  type SandboxCallbacks
} from './sandbox-state';

/** Saved documents contain references/configuration; export is an explicitly requested GPU readback. */
export function SandboxPanel(callbacks: SandboxCallbacks & {mode?: 'export' | 'workspace'}) {
  const [saved, setSaved] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const [inspectingState, setInspectingState] = useState(false);
  const [exportDatasetId, setExportDatasetId] = useState(callbacks.runtime.explorer.dataset.id);
  const [exportFormat, setExportFormat] = useState<
    | 'prepared-json'
    | 'arrow'
    | 'csv'
    | 'analysis-json'
    | 'analysis-arrow'
    | 'analysis-csv'
    | 'combined-arrow'
    | 'combined-csv'
  >('prepared-json');
  const exportDatasets = [
    {id: callbacks.runtime.explorer.dataset.id, label: 'Observations fixture'},
    ...Array.from(callbacks.runtime.publications.values(), published => ({
      id: published.dataset.id,
      label: published.label
    }))
  ];
  const hasExportDataset = exportDatasets.some(dataset => dataset.id === exportDatasetId);
  const cancellation = useRef<AbortController | null>(null);
  useEffect(() => () => cancellation.current?.abort(), [callbacks.runtime]);
  const invoke = async (operation: () => void | Promise<void>) => {
    setBusy(true);
    setStatus('');
    try {
      await operation();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };
  const exportRows = (mode: 'page' | 'all-selected') =>
    invoke(async () => {
      const controller = new AbortController();
      cancellation.current = controller;
      try {
        const explorer = getSandboxExplorer(callbacks.runtime, exportDatasetId);
        const analysis =
          exportDatasetId === callbacks.runtime.explorer.dataset.id
            ? callbacks.runtime.analysis
            : callbacks.runtime.publications.get(exportDatasetId)?.analysis;
        const result =
          exportFormat === 'analysis-json' ||
          exportFormat === 'analysis-arrow' ||
          exportFormat === 'analysis-csv' ||
          exportFormat === 'combined-arrow' ||
          exportFormat === 'combined-csv'
            ? await (() => {
                if (!analysis)
                  throw new Error('Run an analysis on this dataset before exporting results');
                return exportSelectedAnalysis(callbacks.runtime.session, explorer, analysis, {
                  ...(mode === 'page'
                    ? {mode, offset: 0, limit: 100}
                    : {mode, maximumRows: 1_000_000}),
                  format:
                    exportFormat === 'analysis-json'
                      ? 'json'
                      : exportFormat === 'analysis-arrow' || exportFormat === 'combined-arrow'
                        ? 'arrow'
                        : 'csv',
                  includeOriginalSource:
                    exportFormat === 'combined-arrow' || exportFormat === 'combined-csv',
                  maximumReadbackBytes: 64 * 1024 * 1024,
                  maximumCpuBytes: 64 * 1024 * 1024,
                  maximumOutputBytes: 64 * 1024 * 1024,
                  signal: controller.signal
                });
              })()
            : exportFormat === 'prepared-json'
              ? await exportSandboxSelection(
                  callbacks.runtime,
                  mode,
                  controller.signal,
                  exportDatasetId
                )
              : await exportSelectedSourceWithColumnEdits(
                  callbacks.runtime.session,
                  explorer,
                  {
                    ...(mode === 'page'
                      ? {mode, offset: 0, limit: 100}
                      : {mode, maximumRows: 1_000_000}),
                    format: exportFormat,
                    maximumReadbackBytes: 64 * 1024 * 1024,
                    maximumCpuBytes: 64 * 1024 * 1024,
                    maximumOutputBytes: 64 * 1024 * 1024,
                    signal: controller.signal
                  },
                  residentColumnEditStore.get(explorer.dataset.id)
                );
        try {
          controller.signal.throwIfAborted();
          if ('assertCurrent' in result) result.assertCurrent();
          if (
            !explorer.isCurrent ||
            explorer.currentRevision !== result.selectionRevision ||
            explorer.dataset.revision !== result.datasetRevision
          )
            throw new Error('Selection changed before export publication');
          if ('bytes' in result)
            download(
              `kepler-${result.datasetId}-${mode}-${result.datasetRevision}-${exportFormat}.${result.format}`,
              new Blob([result.bytes], {
                type:
                  result.format === 'arrow'
                    ? 'application/vnd.apache.arrow.stream'
                    : result.format === 'json'
                      ? 'application/json'
                      : 'text/csv;charset=utf-8'
              })
            );
          else
            download(
              `kepler-${result.datasetId}-${mode}-${result.datasetRevision}.json`,
              serializeSandboxExport(result)
            );
          if ('metadata' in result && result.format === 'csv')
            download(
              `kepler-${result.datasetId}-${mode}-${result.datasetRevision}-${exportFormat}.metadata.json`,
              result.metadata
            );
          setStatus(
            `${result.rowCount.toLocaleString()} rows exported (${mode}, ${exportFormat}); ${result.readbackBytes.toLocaleString()} GPU bytes read.`
          );
        } finally {
          result.destroy();
        }
      } finally {
        if (cancellation.current === controller) cancellation.current = null;
      }
    });
  return (
    <section
      aria-label={callbacks.mode === 'export' ? 'Export data' : 'Save map'}
      style={{borderTop: '1px solid #405168', marginTop: 18, paddingTop: 10}}
    >
      <h2 style={{fontSize: 16}}>
        {callbacks.mode === 'export' ? 'Export data' : 'Save and restore map'}
      </h2>
      {callbacks.mode !== 'export' && (
        <>
          <p className="hint">
            State files save source references, native map configuration, selections and analysis.
            Uploaded sources need a portable bundle to rebuild them in another session.
          </p>
          <div className="actions">
            <button
              disabled={busy}
              onClick={() =>
                void invoke(() => {
                  const serialized = saveSandbox(callbacks);
                  setSaved(serialized);
                  download('kepler-sandbox-state.json', serialized);
                  setStatus('Sandbox state saved.');
                })
              }
            >
              Save state
            </button>
            <label>
              Load state{' '}
              <input
                aria-label="Load sandbox state"
                type="file"
                accept="application/json,.json"
                disabled={busy}
                onChange={event => {
                  const file = event.target.files?.[0];
                  if (file)
                    void invoke(async () => {
                      if (file.size > 4 * 1024 * 1024) throw new Error('State file exceeds 4 MiB');
                      setSaved(await file.text());
                      setStatus('State loaded; choose replay or reimport.');
                    });
                }}
              />
            </label>
          </div>
          <details onToggle={event => setInspectingState(event.currentTarget.open)}>
            <summary>Inspect or edit saved state</summary>
            <textarea
              aria-label="Saved sandbox state"
              // A textarea in a closed <details> is not focusable; keep it out of the dialog's
              // Tab scope (react-modal's Safari path would otherwise try to focus it and stall).
              tabIndex={inspectingState ? 0 : -1}
              rows={6}
              value={saved}
              onChange={event => setSaved(event.target.value)}
              style={{width: '100%', boxSizing: 'border-box'}}
            />
          </details>
          <div className="actions">
            <button
              disabled={busy || !saved}
              onClick={() =>
                void invoke(async () => {
                  await replaySandbox(saved, callbacks);
                  setStatus('Native configuration, selections and analysis replayed.');
                })
              }
            >
              Replay state
            </button>
            {callbacks.reimportSource ? (
              <button
                disabled={busy || !saved}
                onClick={() =>
                  void invoke(async () => {
                    await replaySandbox(saved, callbacks, true);
                    setStatus('Source regenerated and saved state replayed.');
                  })
                }
              >
                Reimport and replay
              </button>
            ) : null}
          </div>
          <SandboxBundlePanel callbacks={callbacks} disabled={busy} onBusyChange={setBusy} />
          <MediaExportPanel callbacks={callbacks} disabled={busy} onBusyChange={setBusy} />
          <SandboxSharePanel
            disabled={busy}
            saveState={() => saveSandbox(callbacks)}
            loadState={setSaved}
          />
          <ShareLinkSourcesPanel disabled={busy} />
        </>
      )}
      {callbacks.mode !== 'workspace' && (
        <>
          <label>
            Export dataset
            <select
              aria-label="Export dataset"
              value={hasExportDataset ? exportDatasetId : ''}
              disabled={busy}
              onChange={event => setExportDatasetId(event.target.value)}
            >
              <option value="">Choose a dataset</option>
              {exportDatasets.map(dataset => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Export format
            <select
              aria-label="Export format"
              value={exportFormat}
              disabled={busy}
              onChange={event => {
                const value = event.target.value;
                if (
                  value === 'prepared-json' ||
                  value === 'arrow' ||
                  value === 'csv' ||
                  value === 'analysis-json' ||
                  value === 'analysis-arrow' ||
                  value === 'analysis-csv' ||
                  value === 'combined-arrow' ||
                  value === 'combined-csv'
                )
                  setExportFormat(value);
              }}
            >
              <option value="prepared-json">Prepared GPU values · JSON</option>
              <option value="arrow">Original source values · Arrow</option>
              <option value="csv">Original source values · CSV</option>
              <option value="analysis-json">Analysis results · JSON</option>
              <option value="analysis-arrow">Analysis results · Arrow</option>
              <option value="analysis-csv">Analysis results · CSV + metadata</option>
              <option value="combined-arrow">Original + analysis fields · Arrow</option>
              <option value="combined-csv">Original + analysis fields · CSV + metadata</option>
            </select>
          </label>
          <p className="hint">
            Arrow and CSV use GPU-selected row identities to export retained source values. Arrow
            keeps exact types, timestamps, geometry and dictionary domains; CSV writes text with UTC
            timestamps and JSON geometry. Every format includes original row identities. Source
            exports exclude derived GPU fields. Analysis exports contain the current computed field,
            nulls, source-row identities and its recipe. Combined exports join both by original row
            identity; colliding analysis names get an _analysis suffix. CSV analysis exports include
            a separate metadata file.
          </p>
          <p className="hint">
            Up to 1,000,000 selected rows and 64 MiB GPU readback. Source downloads allow 64 MiB of
            output and reserve up to 128 MiB for serialization; JavaScript metadata uses additional
            memory.
          </p>
          <div className="actions">
            <button disabled={busy || !hasExportDataset} onClick={() => void exportRows('page')}>
              Export 100-row page
            </button>
            <button
              disabled={busy || !hasExportDataset}
              onClick={() => void exportRows('all-selected')}
            >
              Export all selected
            </button>
            {busy && cancellation.current ? (
              <button onClick={() => cancellation.current?.abort()}>Cancel export</button>
            ) : null}
          </div>
          <StreamingExportPanel runtime={callbacks.runtime} disabled={busy} />
        </>
      )}
      {status ? (
        <p role="status" className="hint">
          {status}
        </p>
      ) : null}
    </section>
  );
}

function download(filename: string, content: string | Blob): void {
  const url = URL.createObjectURL(
    typeof content === 'string' ? new Blob([content], {type: 'application/json'}) : content
  );
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
