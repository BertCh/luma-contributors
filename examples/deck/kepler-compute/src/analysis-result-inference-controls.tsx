// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {
  applyKeplerResultInference,
  getCatalogBinding,
  getKeplerResultInferenceSettings,
  hasKeplerResultInference,
  resolveKeplerOperationDescriptor,
  type KeplerResultInferenceScope,
  type KeplerResultInferenceSettings,
  type KeplerResultInferenceUpdate
} from '@luma-contributors/kepler';
import type {AnalysisResultCardModel} from './analysis-result-card-model';

/** Hosts accept the whole display before the card commits a new sentence or legend. */
export function AnalysisResultInferenceControls({
  model,
  scope = 'full',
  onApply,
  onRerun,
  disabled = false
}: {
  model: AnalysisResultCardModel;
  scope?: KeplerResultInferenceScope;
  onApply?(update: KeplerResultInferenceUpdate): void | Promise<void>;
  onRerun?(settings: KeplerResultInferenceSettings): void | Promise<void>;
  disabled?: boolean;
}) {
  const [settings, setSettings] = useState(() => getKeplerResultInferenceSettings(model.result));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  // Non-inferential results do not acquire meaningless significance controls.
  const statistical = hasKeplerResultInference(model.report, model.presentations);
  if (!statistical) return null;
  const binding = getCatalogBinding(model.result.operationId)!;
  const {descriptor} = resolveKeplerOperationDescriptor(binding);
  const update = applyKeplerResultInference(
    model.result,
    binding,
    descriptor,
    settings,
    model.context,
    scope
  );
  async function apply() {
    setBusy(true);
    setError('');
    try {
      if (update.status === 'display') await onApply?.(update);
      else await onRerun?.(settings);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }
  return (
    <fieldset data-testid="ax-inference-controls" disabled={busy || disabled}>
      <legend>Inference display</legend>
      <label>
        Significance cut-off{' '}
        <select
          aria-label="Significance cut-off"
          value={settings.significance}
          onChange={event => setSettings({...settings, significance: Number(event.target.value)})}
        >
          <option value={0.1}>0.10 (90%)</option>
          <option value={0.05}>0.05 (95%)</option>
          <option value={0.01}>0.01 (99%)</option>
          {![0.1, 0.05, 0.01].includes(settings.significance) && (
            <option value={settings.significance}>
              {settings.significance} (executed cut-off)
            </option>
          )}
        </select>
      </label>{' '}
      <label>
        Correction{' '}
        <select
          aria-label="Correction"
          value={settings.multipleTesting}
          onChange={event =>
            setSettings({
              ...settings,
              multipleTesting: event.target
                .value as KeplerResultInferenceSettings['multipleTesting']
            })
          }
        >
          <option value="benjamini-hochberg">Benjamini-Hochberg (FDR)</option>
          {model.result.operationId !== 'recipe-hot-spot-lattice' && (
            <option value="bonferroni">Bonferroni</option>
          )}
          <option value="none">None</option>
        </select>
      </label>
      <p>
        {update.status === 'display'
          ? 'Uses the retained p-values across the full tested population. The statistic will not rerun.'
          : `Rerun required: ${update.reason}`}
      </p>
      {update.status === 'display' && !onApply && (
        <p>
          Display changes are unavailable in this view. Reopen the analysis settings to change
          inference.
        </p>
      )}
      {update.status === 'rerun-required' && !onRerun && (
        <p>Reopen the analysis settings and run with the requested inference settings.</p>
      )}
      <button
        type="button"
        onClick={apply}
        disabled={update.status === 'display' ? !onApply : !onRerun}
      >
        {busy ? 'Applying…' : update.status === 'display' ? 'Apply to display' : 'Rerun analysis'}
      </button>
      {error && <p role="alert">{error}</p>}
    </fieldset>
  );
}
