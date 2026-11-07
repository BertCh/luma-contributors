// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useRef, useState} from 'react';
import {getApplicationConfig, initApplicationConfig} from '@kepler.gl/utils';
import {selectEmbeddedIconCatalog} from './icon-catalog-assets';
import type {NativeIconDefinition} from './icon-source';
import {collectNativeOnlySources, describeSkippedNativeSources} from './bundle-native-sources';
import {getNativeOnlyDatasets} from './native-source-access';
import {
  parseSandboxBundle,
  serializeSandboxBundle,
  type RestoredSandboxBundle,
  type SandboxBundleLimits
} from './sandbox-bundle';
import {
  materializeSandboxBundleSources,
  replaySandboxBundle,
  saveSandbox,
  validateSandboxBundle,
  type SandboxCallbacks
} from './sandbox-state';

const BUNDLE_LIMITS: SandboxBundleLimits = {
  maximumDocumentBytes: 96 * 1024 * 1024,
  maximumSourceBytes: 64 * 1024 * 1024,
  maximumRows: 1_000_000,
  maximumSources: 16
};

export function SandboxBundlePanel({
  callbacks,
  disabled,
  onBusyChange
}: {
  callbacks: SandboxCallbacks;
  disabled: boolean;
  onBusyChange(busy: boolean): void;
}) {
  const [bundle, setBundle] = useState<RestoredSandboxBundle | null>(null);
  const [status, setStatus] = useState('');
  const pending = useRef(false);
  const invoke = async (action: () => Promise<void>) => {
    if (disabled || pending.current) return;
    pending.current = true;
    onBusyChange(true);
    setStatus('');
    try {
      await action();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      pending.current = false;
      onBusyChange(false);
    }
  };
  return (
    <section aria-label="Portable sandbox bundle">
      <h3>Portable workspace</h3>
      <p className="hint">
        Include full data for resident datasets and SQL results with map settings. Export uses
        retained Arrow data, including rows outside the current selection. Partial stream
        populations are saved as the admitted prefix and restored as partial.
      </p>
      <div className="actions">
        <button
          disabled={disabled}
          onClick={() =>
            void invoke(async () => {
              // Datasets that never reached the GPU carry their original native values.
              const nativeOnly = await collectNativeOnlySources(getNativeOnlyDatasets());
              const skippedMessage = describeSkippedNativeSources(nativeOnly.skipped);
              let state: string;
              try {
                state = saveSandbox(
                  callbacks,
                  nativeOnly.embedded.map(source => source.datasetId)
                );
              } catch (error) {
                throw new Error(
                  `${error instanceof Error ? error.message : String(error)} ${skippedMessage}`.trim()
                );
              }
              const serialized = await serializeSandboxBundle(
                state,
                await materializeSandboxBundleSources(callbacks.runtime, BUNDLE_LIMITS),
                BUNDLE_LIMITS,
                selectEmbeddedIconCatalog(
                  // Native application config types custom icons with a mutable mesh shape.
                  getApplicationConfig().customIcons as NativeIconDefinition[]
                ),
                nativeOnly.embedded
              );
              const url = URL.createObjectURL(new Blob([serialized], {type: 'application/json'}));
              try {
                const anchor = document.createElement('a');
                anchor.href = url;
                anchor.download = 'kepler-workspace.bundle.json';
                anchor.click();
              } finally {
                setTimeout(() => URL.revokeObjectURL(url), 0);
              }
              const partialLabels = Array.from(
                callbacks.runtime.publications.values(),
                published => published.dataset.population?.partial && published.label
              ).filter(Boolean);
              const nativeNote = nativeOnly.embedded.length
                ? ` Native-only datasets embedded: ${nativeOnly.embedded.map(source => source.label).join(', ')}.`
                : '';
              setStatus(
                `${
                  partialLabels.length
                    ? `Portable workspace saved. Partial populations (incomplete prefixes, not the full source): ${partialLabels.join('; ')}.`
                    : 'Portable workspace saved with its full source data.'
                }${nativeNote} ${skippedMessage}`.trim()
              );
            })
          }
        >
          Save portable bundle
        </button>
        {callbacks.restoreBundle ? (
          <label>
            Load portable bundle
            <input
              aria-label="Load portable bundle"
              type="file"
              accept="application/json,.json"
              disabled={disabled}
              onChange={event => {
                const file = event.target.files?.[0];
                event.target.value = '';
                if (!file) return;
                setBundle(null);
                void invoke(async () => {
                  if (file.size > BUNDLE_LIMITS.maximumDocumentBytes)
                    throw new Error('Bundle file exceeds 96 MiB');
                  const restored = await parseSandboxBundle(await file.text(), BUNDLE_LIMITS);
                  validateSandboxBundle(restored, callbacks.runtime);
                  setBundle(restored);
                  setStatus('Bundle validated. Review its sources before restoring the workspace.');
                });
              }}
            />
          </label>
        ) : null}
      </div>
      {bundle ? (
        <>
          <ul>
            {bundle.sources.map(source => {
              const descriptor = bundle.state.sources.find(
                item => item.dataset.datasetId === source.datasetId
              )!;
              return (
                <li key={source.datasetId}>
                  {source.datasetId}: {source.table.numRows.toLocaleString()} rows,{' '}
                  {source.table.batches.length} batches, {descriptor.schema.length} fields
                </li>
              );
            })}
            {(bundle.nativeOnlySources ?? []).map(source => (
              <li key={source.datasetId}>
                {source.datasetId}: {source.table.numRows.toLocaleString()} rows (native-only
                dataset, restored as a native table)
              </li>
            ))}
          </ul>
          <p className="hint">
            Restore replaces the current workspace with these sources and replays the saved map,
            filters and analysis.
          </p>
          <button
            disabled={disabled}
            onClick={() =>
              void invoke(async () => {
                // Embedded custom icons must be installed before replayed icon layers load.
                if (bundle.iconCatalog)
                  initApplicationConfig({customIcons: bundle.iconCatalog as never});
                await replaySandboxBundle(bundle, callbacks);
                setBundle(null);
                setStatus('Portable workspace restored and saved state replayed.');
              })
            }
          >
            Restore portable workspace
          </button>
        </>
      ) : null}
      <p className="hint">
        Up to 16 sources, 1,000,000 rows per source, 64 MiB Arrow data and 96 MiB JSON. Native-only
        datasets (never on the GPU) add up to 8 tables and 32 MiB of Arrow data; larger ones are
        reported and not embedded. Serialization and parsing require additional CPU memory.
      </p>
      {status ? (
        <p role="status" className="hint">
          {status}
        </p>
      ) : null}
    </section>
  );
}
