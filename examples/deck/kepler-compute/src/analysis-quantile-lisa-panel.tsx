// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS} from '@luma-contributors/kepler';
import {
  normalizeQuantileLisaSettings,
  type QuantileLisaSettings
} from './analysis-quantile-lisa-settings';
import {PointTopologyControls, usePointTopology} from './analysis-point-topology-controls';
import {PointCoordinateControls, usePointCoordinates} from './analysis-point-coordinate-controls';
import {validatePointCoordinateColumns} from './analysis-point-coordinate-settings';
import {ResidentConsumerNotice} from './resident-consumer-notice';

const NUMERIC_FIELDS = [
  {key: 'quantileCount', label: 'Quantile LISA quantile count', minimum: 2, step: '1'},
  {key: 'selectedQuantile', label: 'Quantile LISA selected quantile', minimum: 1, step: '1'},
  {
    key: 'seed',
    label: 'Quantile LISA seed',
    minimum: 0,
    maximum: Number.MAX_SAFE_INTEGER,
    step: '1'
  },
  {
    key: 'permutationCount',
    label: 'Quantile LISA permutations',
    minimum: 1,
    maximum: MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS,
    step: '1'
  },
  {
    key: 'significanceLevel',
    label: 'Quantile LISA significance level',
    minimum: 0,
    maximum: 1,
    step: 'any'
  }
] as const;

function makeDraft(settings?: QuantileLisaSettings) {
  return {
    quantileCount: String(settings?.quantileCount ?? 2),
    selectedQuantile: String(settings?.selectedQuantile ?? 2),
    seed: String(settings?.seed ?? 314159),
    permutationCount: String(settings?.permutationCount ?? 199),
    significanceLevel: String(settings?.significanceLevel ?? 0.05)
  };
}

export function QuantileLisaPanel({
  columns,
  rowCount,
  settings,
  onApply
}: {
  columns: readonly string[];
  rowCount: number;
  settings?: QuantileLisaSettings;
  onApply(settings: QuantileLisaSettings): void;
}) {
  const [draft, setDraft] = useState(() => makeDraft(settings));
  const coordinates = usePointCoordinates(settings);
  const topology = usePointTopology(settings, rowCount);
  const [error, setError] = useState('');
  useEffect(() => {
    setDraft(makeDraft(settings));
    setError('');
  }, [
    settings?.quantileCount,
    settings?.selectedQuantile,
    settings?.seed,
    settings?.permutationCount,
    settings?.significanceLevel
  ]);
  return (
    <details className="analysis-quantile-lisa-settings">
      <summary>
        Configure Quantile LISA on{' '}
        {coordinates.draft.coordinateSpace === 'planar' ? 'projected' : 'geographic'} points
      </summary>
      <p className="hint">
        Choose one quantile of the numeric field and count neighboring points in that same quantile.
        Choose geographic or projected meter coordinates; spatial weights are binary.
      </p>
      <form
        onSubmit={event => {
          event.preventDefault();
          try {
            const coordinateSettings = coordinates.getSettings();
            validatePointCoordinateColumns(coordinateSettings, columns);
            onApply(
              normalizeQuantileLisaSettings(
                {
                  ...coordinateSettings,
                  ...topology.getSettings(),
                  quantileCount: Number(draft.quantileCount),
                  selectedQuantile: Number(draft.selectedQuantile),
                  seed: Number(draft.seed),
                  permutationCount: Number(draft.permutationCount),
                  significanceLevel: Number(draft.significanceLevel)
                },
                rowCount
              )
            );
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        <PointCoordinateControls
          value={coordinates.draft}
          onChange={coordinates.setDraft}
          columns={columns}
          idPrefix="analysis-quantile-lisa"
          labelPrefix="Quantile LISA"
          longitudeId="analysis-quantile-lisa-longitudeColumn"
          latitudeId="analysis-quantile-lisa-latitudeColumn"
        />
        <PointTopologyControls
          topology={topology}
          coordinateSpace={coordinates.draft.coordinateSpace}
          idPrefix="analysis-quantile-lisa"
          labelPrefix="Quantile LISA"
          nearestLabel="Quantile LISA nearest neighbors"
          nearestId="analysis-quantile-lisa-k"
          permutationMaximumNeighbors={64}
        />
        {NUMERIC_FIELDS.map(field => (
          <label key={field.key} htmlFor={`analysis-quantile-lisa-${field.key}`}>
            {field.label}
            <input
              id={`analysis-quantile-lisa-${field.key}`}
              type="number"
              required
              min={field.minimum}
              max={
                'maximum' in field
                  ? field.maximum
                  : field.key === 'quantileCount'
                    ? rowCount - 1
                    : Number(draft.quantileCount)
              }
              step={field.step}
              value={draft[field.key]}
              onChange={event => setDraft({...draft, [field.key]: event.target.value})}
            />
          </label>
        ))}
        <p className="hint">
          Type-5 quantiles include the lower boundary and exclude the upper boundary. Ties stay
          together and can leave quantiles empty. Missing numeric values and invalid coordinates are
          omitted before quantiling. Selection recomputes boundaries and removes neighbors from the
          fixed source graph without replacements.
        </p>
        <p className="hint">
          This compatibility statistic uses legacy directed pseudo p-values: (min(upper-tail count,
          permutations − upper-tail count) + 1) / (permutations + 1). It is anti-conservative, has
          no two-sided error-control guarantee and applies no multiple-testing correction. Tied or
          constant memberships can appear significant. Treat results as exploratory.
        </p>
        <p className="hint">
          Only selected-quantile members with positive local joins are tested. Isolated points are
          excluded from inference. Gray includes nonsignificant and untested rows; red denotes
          significance under the legacy rule; orange denotes isolated points. The table retains
          membership, join count, pseudo p-value and legacy tail count. A fixed seed is reproducible
          for the same source and selection. More rows, neighbors and permutations increase
          computation time.
        </p>
        <button type="submit" disabled={rowCount < 3}>
          Apply neighbors and run Quantile LISA
        </button>
      </form>
      {error && (
        <ResidentConsumerNotice
          consumer="analysis"
          status="error"
          className="error"
          reason={error}
        />
      )}
    </details>
  );
}
