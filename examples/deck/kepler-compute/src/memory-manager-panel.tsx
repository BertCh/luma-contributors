// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {formatMemoryBytes, type MemoryManagerModel} from './memory-manager-model';

export type MemoryManagerSource = {
  /** Re-renders on registry memory, eviction status or layer changes. */
  subscribe(listener: () => void): () => void;
  getModel(): MemoryManagerModel | null;
  unloadDataset(datasetId: string): void;
  evictDataset(datasetId: string): void;
  /** Rejects when the new budget is below current usage. */
  setBudgetMebibytes(mebibytes: number): string | null;
  isEvictionEnabled: boolean;
  /** Opens the full Kepler modal; omitted inside the modal itself. */
  openManager?(): void;
};

const BUDGET_PRESETS_MEBIBYTES = [256, 512, 1024, 2048, 4096] as const;

/**
 * Native-surface memory manager. The compact form sits in the Kepler side panel; the full form
 * renders in a Kepler modal with per-dataset and per-layer accounting and unload, evict and
 * budget actions. All figures come from the session registry; nothing is estimated here.
 */
export function MemoryManagerPanel({
  source,
  variant
}: {
  source: MemoryManagerSource;
  variant: 'compact' | 'full';
}) {
  const [, setVersion] = useState(0);
  const [budgetText, setBudgetText] = useState('');
  const [budgetError, setBudgetError] = useState<string | null>(null);
  useEffect(() => source.subscribe(() => setVersion(value => value + 1)), [source]);
  const model = source.getModel();
  if (!model) return null;
  const percent = Math.round(model.usedFraction * 100);
  return (
    <section aria-label="Memory manager" data-testid={`memory-manager-${variant}`}>
      <h2>Memory</h2>
      <meter
        aria-label="Memory used"
        min={0}
        max={model.budgetBytes}
        low={model.budgetBytes * 0.8}
        high={model.budgetBytes * 0.95}
        optimum={0}
        value={Math.min(model.budgetBytes, model.usedBytes)}
        data-testid="memory-manager-meter"
      />
      <p className="hint" data-testid="memory-manager-summary" data-pressure={model.pressure}>
        {percent}% used: {formatMemoryBytes(model.usedBytes)} of{' '}
        {formatMemoryBytes(model.budgetBytes)}; {formatMemoryBytes(model.availableBytes)} available
      </p>
      {model.pressure !== 'ok' ? (
        <p role="alert" className="error" data-testid="memory-manager-warning">
          {model.pressure === 'critical'
            ? 'Memory is nearly exhausted; new requests may be refused.'
            : 'Memory use is high.'}
        </p>
      ) : null}
      <ul data-testid="memory-manager-operations" aria-label="Running operations">
        {model.operations.map(operation => (
          <li key={operation.key} data-operation-kind={operation.kind}>
            {operation.label}
            {operation.detail ? ` - ${operation.detail}` : ''}
            <progress
              aria-label={`${operation.label} progress`}
              max={1}
              {...(operation.fraction === undefined ? {} : {value: operation.fraction})}
            />
            <button type="button" onClick={() => operation.cancel()}>
              Cancel
            </button>
          </li>
        ))}
        {model.operations.length === 0 ? <li className="hint">No running operations.</li> : null}
      </ul>
      {variant === 'compact' ? (
        <>
          <p className="hint">
            {model.datasets.length} datasets, {model.layers.length} layers. Largest:{' '}
            {[...model.datasets]
              .sort((a, b) => b.cpuBytes + b.gpuBytes - (a.cpuBytes + a.gpuBytes))
              .slice(0, 2)
              .map(row => `${row.label} ${formatMemoryBytes(row.cpuBytes + row.gpuBytes)}`)
              .join(', ') || 'none'}
          </p>
          {source.openManager ? (
            <button type="button" data-testid="memory-manager-open" onClick={source.openManager}>
              Open memory manager
            </button>
          ) : null}
        </>
      ) : (
        <>
          <h3>Datasets</h3>
          <table data-testid="memory-manager-datasets" aria-label="Datasets">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Rows</th>
                <th>GPU</th>
                <th>CPU</th>
                <th>Frees</th>
                <th>Layers</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {model.datasets.map(row => (
                <tr
                  key={row.id}
                  data-dataset-id={row.id}
                  data-evicted={row.isEvicted}
                  data-releasable-bytes={row.releasableBytes}
                >
                  <td>
                    {row.label}
                    {row.isPinned ? ' (pinned)' : ''}
                    {row.isEvicted ? ' (evicted)' : ''}
                  </td>
                  <td>{row.rowCount.toLocaleString()}</td>
                  <td>{formatMemoryBytes(row.gpuBytes)}</td>
                  <td>{formatMemoryBytes(row.cpuBytes)}</td>
                  <td>{formatMemoryBytes(row.releasableBytes)}</td>
                  <td>{row.layerIds.length}</td>
                  <td>
                    <button
                      type="button"
                      disabled={!row.canEvict}
                      title={
                        source.isEvictionEnabled
                          ? 'Release now; reloads from its retained source when used'
                          : 'Needs ?automatic=evict'
                      }
                      onClick={() => source.evictDataset(row.id)}
                    >
                      Evict
                    </button>
                    <button
                      type="button"
                      disabled={!row.canUnload}
                      title="Remove the dataset, its layers and filters"
                      onClick={() => source.unloadDataset(row.id)}
                    >
                      Unload
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>Layers</h3>
          <table data-testid="memory-manager-layers" aria-label="Layers">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Dataset</th>
                <th>Render buffers</th>
                <th>Dataset</th>
                <th>Shared with</th>
              </tr>
            </thead>
            <tbody>
              {model.layers.map(row => (
                <tr key={row.id} data-layer-id={row.id}>
                  <td>
                    {row.label}
                    {row.isVisible ? '' : ' (hidden)'}
                  </td>
                  <td>{row.dataId ?? 'none'}</td>
                  <td data-render-buffer-bytes={row.renderBufferBytes}>
                    {row.isMeasured
                      ? `${formatMemoryBytes(row.renderBufferBytes)} (${row.renderBufferCount} buffers)`
                      : 'not rendered'}
                  </td>
                  <td>{formatMemoryBytes(row.datasetBytes)}</td>
                  <td>{row.sharedWithLayers > 1 ? `${row.sharedWithLayers} layers` : 'no one'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint">
            Render buffers are measured from the buffers each layer's deck.gl layers allocated
            themselves ({formatMemoryBytes(model.layerRenderBufferBytes)} in total, outside the
            budget); registry data a layer only borrows is counted once, under its dataset.
            Reservations: source {formatMemoryBytes(model.reservations.source)}, output{' '}
            {formatMemoryBytes(model.reservations.output)}, scratch{' '}
            {formatMemoryBytes(model.reservations.scratch)}, readback{' '}
            {formatMemoryBytes(model.reservations.readback)}; unattributed resident{' '}
            {formatMemoryBytes(model.unattributedBytes)}.
          </p>
          <h3>Budget</h3>
          <div className="actions">
            {BUDGET_PRESETS_MEBIBYTES.map(mebibytes => (
              <button
                key={mebibytes}
                type="button"
                aria-pressed={model.budgetBytes === mebibytes * 1024 * 1024}
                onClick={() => setBudgetError(source.setBudgetMebibytes(mebibytes))}
              >
                {mebibytes >= 1024 ? `${mebibytes / 1024} GiB` : `${mebibytes} MiB`}
              </button>
            ))}
          </div>
          <label>
            Custom budget (MiB)
            <input
              type="number"
              min={1}
              step={1}
              value={budgetText}
              data-testid="memory-manager-budget-input"
              onChange={event => setBudgetText(event.target.value)}
            />
          </label>
          <button
            type="button"
            data-testid="memory-manager-budget-apply"
            disabled={!Number.isInteger(Number(budgetText)) || Number(budgetText) < 1}
            onClick={() => setBudgetError(source.setBudgetMebibytes(Number(budgetText)))}
          >
            Apply budget
          </button>
          {budgetError ? (
            <p role="alert" className="error" data-testid="memory-manager-budget-error">
              {budgetError}
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}
