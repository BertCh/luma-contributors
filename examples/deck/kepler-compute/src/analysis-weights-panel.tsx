// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import type {AnalysisRuntime} from './analysis-runtime';
import type {AnalysisWeightsInspection, AnalysisWeightsSettings} from './analysis-weights-store';

/** Shared by point and polygon forms; topology settings remain in their existing controls. */
export function AnalysisWeightsPanel({
  analysis,
  family,
  getSettings,
  draftDisabled = false
}: {
  analysis: AnalysisRuntime;
  family: 'point' | 'polygon';
  getSettings(): AnalysisWeightsSettings;
  /** Only actions which consume draft settings are gated by the owning form. */
  draftDisabled?: boolean;
}) {
  const store = analysis.weightsStore;
  const [, setRevision] = useState(0);
  useEffect(() => store.subscribe(() => setRevision(revision => revision + 1)), [store]);
  const definitions = store
    .list(analysis.namedWeightsSource)
    .filter(definition => (definition.settings.kind === 'contiguity') === (family === 'polygon'));
  const reference = analysis.weightsReference;
  const selected = definitions.find(definition => definition.id === reference?.weightsId);
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [inspection, setInspection] = useState<AnalysisWeightsInspection>();
  useEffect(() => {
    setName(selected?.name ?? '');
    setInspection(undefined);
  }, [selected?.id, selected?.name, selected?.version]);
  const act = (action: () => void) => {
    try {
      action();
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };
  const prefix = `analysis-${family}-named-weights`;
  const currentInspection =
    inspection?.weightsId === selected?.id && inspection?.version === selected?.version
      ? inspection
      : undefined;
  return (
    <fieldset className="analysis-named-weights">
      <legend>Reusable spatial weights</legend>
      <label htmlFor={`${prefix}-select`}>Use weights</label>
      <select
        id={`${prefix}-select`}
        value={selected?.id ?? ''}
        onChange={event => act(() => analysis.selectNamedWeights(event.target.value))}
      >
        <option value="" disabled>
          Apply an analysis, then choose its weights
        </option>
        {definitions.map(definition => (
          <option key={definition.id} value={definition.id}>
            {definition.name} · v{definition.version}
          </option>
        ))}
      </select>
      <label htmlFor={`${prefix}-name`}>Weights name</label>
      <input id={`${prefix}-name`} value={name} onChange={event => setName(event.target.value)} />
      <button
        type="button"
        disabled={draftDisabled}
        onClick={() =>
          act(() => {
            store.create(analysis.namedWeightsSource, getSettings(), name);
          })
        }
      >
        Save draft as new weights
      </button>
      {selected && (
        <>
          <p>
            {selected.name} · {selected.id} · version {selected.version}
          </p>
          <button type="button" onClick={() => act(() => store.rename(selected.id, name))}>
            Rename weights
          </button>
          <button
            type="button"
            disabled={draftDisabled}
            onClick={() => act(() => store.edit(selected.id, getSettings()))}
          >
            Replace weights with draft settings
          </button>
          <p className="hint">
            Replacing weights marks every dependent analysis stale. Choose “Use current version” in
            each analysis to rebuild.
          </p>
          {reference?.staleReason && <p role="status">Stale: {reference.staleReason}</p>}
          <button type="button" onClick={() => act(() => analysis.selectNamedWeights(selected.id))}>
            Use current version
          </button>
          <ul aria-label="Weights consumers">
            {store.getConsumers(selected.id).map(consumer => (
              <li key={consumer.id}>
                {consumer.id} · v{consumer.version} ·{' '}
                {consumer.staleReason ? `stale: ${consumer.staleReason}` : 'current'}
              </li>
            ))}
          </ul>
          <button
            type="button"
            disabled={pending}
            onClick={() => {
              setPending(true);
              setError('');
              void store
                .inspect(selected.id)
                .then(setInspection)
                .catch(failure =>
                  setError(failure instanceof Error ? failure.message : String(failure))
                )
                .finally(() => setPending(false));
            }}
          >
            {pending ? 'Inspecting weights…' : 'Inspect neighbors'}
          </button>
          {currentInspection && (
            <div aria-label="Weights inspector">
              <p>
                {currentInspection.islands} islands (no active neighbors). Neighbor counts:{' '}
                {currentInspection.minimumNeighbors} to {currentInspection.maximumNeighbors}.
              </p>
              <table>
                <caption>Neighbor-count histogram · full source population</caption>
                <thead>
                  <tr>
                    <th>Neighbors</th>
                    <th>Rows</th>
                    <th>Frequency</th>
                  </tr>
                </thead>
                <tbody>
                  {currentInspection.histogram.map((bin, index) => (
                    <tr key={index}>
                      <td>
                        {Math.ceil(bin.minimum)}–{Math.ceil(bin.maximumExclusive) - 1}
                      </td>
                      <td>{bin.count}</td>
                      <td>
                        <meter min={0} max={analysis.rowCount} value={bin.count}>
                          {bin.count}
                        </meter>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="hint">
                Includes all source rows; excluded geometry or coordinates have no links and count
                as islands. Filtering does not refill their neighborhoods.
              </p>
            </div>
          )}
        </>
      )}
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
    </fieldset>
  );
}
