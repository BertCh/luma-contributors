// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {
  MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS,
  MAXIMUM_KEPLER_KRIGING_CELLS,
  MAXIMUM_KEPLER_KRIGING_GRID_DIMENSION,
  MAXIMUM_KEPLER_KRIGING_NEIGHBORS,
  parseKeplerKrigingSettings,
  type KeplerKrigingSettings
} from '@luma-contributors/kepler';
import type {AnalysisRuntime} from './analysis-runtime';
import type {KrigingVariogramSuggestion} from './kriging-diagnostics';
import {ResidentConsumerNotice} from './resident-consumer-notice';

const MODELS = ['spherical', 'exponential', 'gaussian', 'matern-3/2', 'matern-5/2'] as const;
const NUMERIC_FIELDS = [
  {name: 'west', label: 'Kriging west longitude', minimum: -180, maximum: 180, step: 'any'},
  {name: 'south', label: 'Kriging south latitude', minimum: -90, maximum: 90, step: 'any'},
  {name: 'east', label: 'Kriging east longitude', minimum: -180, maximum: 180, step: 'any'},
  {name: 'north', label: 'Kriging north latitude', minimum: -90, maximum: 90, step: 'any'},
  {
    name: 'width',
    label: 'Kriging columns',
    minimum: 1,
    maximum: MAXIMUM_KEPLER_KRIGING_GRID_DIMENSION,
    step: '1'
  },
  {
    name: 'height',
    label: 'Kriging rows',
    minimum: 1,
    maximum: MAXIMUM_KEPLER_KRIGING_GRID_DIMENSION,
    step: '1'
  },
  {name: 'searchRadiusMeters', label: 'Kriging search radius in meters', minimum: 0, step: 'any'},
  {
    name: 'neighborCount',
    label: 'Kriging nearest samples',
    minimum: 1,
    maximum: MAXIMUM_KEPLER_KRIGING_NEIGHBORS,
    step: '1'
  },
  {
    name: 'minimumNeighbors',
    label: 'Kriging minimum distinct positions',
    minimum: 1,
    maximum: MAXIMUM_KEPLER_KRIGING_NEIGHBORS,
    step: '1'
  },
  {name: 'nugget', label: 'Kriging nugget in squared response units', minimum: 0, step: 'any'},
  {name: 'sill', label: 'Kriging partial sill in squared response units', minimum: 0, step: 'any'},
  {name: 'rangeMeters', label: 'Kriging effective range in meters', minimum: 0, step: 'any'}
] as const;

function makeDraft(settings?: KeplerKrigingSettings) {
  return {
    longitudeColumn: settings?.longitudeColumn ?? '',
    latitudeColumn: settings?.latitudeColumn ?? '',
    west: String(settings?.grid.bounds[0] ?? -74.1),
    south: String(settings?.grid.bounds[1] ?? 40.65),
    east: String(settings?.grid.bounds[2] ?? -73.9),
    north: String(settings?.grid.bounds[3] ?? 40.85),
    width: String(settings?.grid.width ?? 32),
    height: String(settings?.grid.height ?? 32),
    searchRadiusMeters: String(settings?.searchRadiusMeters ?? 10000),
    neighborCount: String(settings?.neighborCount ?? 8),
    minimumNeighbors: String(settings?.minimumNeighbors ?? 3),
    model: settings?.variogram.model ?? 'spherical',
    nugget: String(settings?.variogram.nugget ?? 0),
    sill: String(settings?.variogram.sill ?? 1),
    rangeMeters: String(settings?.variogram.rangeMeters ?? 10000),
    confidenceLevel: String(settings?.confidenceLevel ?? ''),
    crossValidation: settings?.crossValidation ?? false,
    maximumLagMeters: ''
  };
}

/** Manual ordinary kriging parameters are committed together with a separate cell-space grid. */
export function KrigingPanel({analysis}: {analysis: AnalysisRuntime}) {
  const settings = analysis.settings.kriging;
  const [draft, setDraft] = useState(() => makeDraft(settings));
  const [error, setError] = useState('');
  const [fit, setFit] = useState<KrigingVariogramSuggestion | null>(null);
  const [fitting, setFitting] = useState(false);
  const [fitMethod, setFitMethod] = useState<
    'weighted-least-squares' | 'vecchia-likelihood' | 'vecchia-fisher'
  >('weighted-least-squares');
  useEffect(() => {
    setDraft(makeDraft(settings));
    setFitMethod(settings?.variogramFit?.method ?? 'weighted-least-squares');
    setError('');
    setFit(null);
  }, [
    analysis.datasetId,
    settings?.longitudeColumn,
    settings?.latitudeColumn,
    settings?.grid.bounds[0],
    settings?.grid.bounds[1],
    settings?.grid.bounds[2],
    settings?.grid.bounds[3],
    settings?.grid.width,
    settings?.grid.height,
    settings?.searchRadiusMeters,
    settings?.neighborCount,
    settings?.minimumNeighbors,
    settings?.variogram.model,
    settings?.variogram.nugget,
    settings?.variogram.sill,
    settings?.variogram.rangeMeters,
    settings?.variogramFit?.method
  ]);
  const buildSettings = (): KeplerKrigingSettings => {
    if (
      [draft.longitudeColumn, draft.latitudeColumn].some(
        column => !column || !analysis.availableColumns.includes(column)
      )
    )
      throw new Error('Choose prepared Float32 longitude and latitude fields');
    if (NUMERIC_FIELDS.some(field => !draft[field.name].trim()))
      throw new Error('Declare every kriging grid and model parameter');
    const fitMatches =
      fit &&
      fit.settings.variogram.model === draft.model &&
      String(fit.settings.variogram.nugget) === draft.nugget &&
      String(fit.settings.variogram.sill) === draft.sill &&
      String(fit.settings.variogram.rangeMeters) === draft.rangeMeters;
    return parseKeplerKrigingSettings({
      longitudeColumn: draft.longitudeColumn,
      latitudeColumn: draft.latitudeColumn,
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters',
      grid: {
        bounds: [Number(draft.west), Number(draft.south), Number(draft.east), Number(draft.north)],
        width: Number(draft.width),
        height: Number(draft.height)
      },
      searchRadiusMeters: Number(draft.searchRadiusMeters),
      neighborCount: Number(draft.neighborCount),
      minimumNeighbors: Number(draft.minimumNeighbors),
      variogram: {
        model: draft.model,
        nugget: Number(draft.nugget),
        sill: Number(draft.sill),
        rangeMeters: Number(draft.rangeMeters)
      },
      ...(draft.confidenceLevel.trim() ? {confidenceLevel: Number(draft.confidenceLevel)} : {}),
      ...(draft.crossValidation ? {crossValidation: true} : {}),
      ...(fitMatches && fit ? {variogramFit: fit.settings.variogramFit} : {})
    });
  };
  return (
    <details className="analysis-kriging-settings">
      <summary>Configure ordinary kriging grid</summary>
      <p className="hint">
        Predict the numeric response field above at the centers of a separate north-up grid.
        Longitude and latitude use degrees; search radius and variogram range use spherical
        great-circle meters. Grid cells are not original source rows.
      </p>
      <form
        onSubmit={event => {
          event.preventDefault();
          try {
            analysis.configureKriging(buildSettings());
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        {(['longitudeColumn', 'latitudeColumn'] as const).map(field => (
          <label key={field} htmlFor={`kriging-${field}`}>
            Kriging {field === 'longitudeColumn' ? 'longitude' : 'latitude'} field
            <select
              id={`kriging-${field}`}
              aria-label={`Kriging ${field === 'longitudeColumn' ? 'longitude' : 'latitude'} field`}
              required
              value={draft[field]}
              onChange={event => setDraft({...draft, [field]: event.target.value})}
            >
              <option value="">Choose a coordinate field</option>
              {analysis.availableColumns.map(column => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </label>
        ))}
        <label htmlFor="kriging-model">Kriging variogram model</label>
        <select
          id="kriging-model"
          value={draft.model}
          onChange={event => {
            const model = MODELS.find(model => model === event.target.value);
            if (model) setDraft({...draft, model});
          }}
        >
          <option value="spherical">Spherical</option>
          <option value="exponential">Exponential</option>
          <option value="gaussian">Gaussian</option>
          <option value="matern-3/2">Matérn 3/2</option>
          <option value="matern-5/2">Matérn 5/2</option>
        </select>
        {NUMERIC_FIELDS.map(field => (
          <label key={field.name} htmlFor={`kriging-${field.name}`}>
            {field.label}
            <input
              id={`kriging-${field.name}`}
              type="number"
              required
              min={field.minimum}
              max={'maximum' in field ? field.maximum : undefined}
              step={field.step}
              value={draft[field.name]}
              onChange={event => setDraft({...draft, [field.name]: event.target.value})}
            />
          </label>
        ))}
        <label htmlFor="kriging-maximumLagMeters">
          Variogram maximum lag in meters (blank: a third of the data extent)
          <input
            id="kriging-maximumLagMeters"
            type="number"
            min={0}
            step="any"
            value={draft.maximumLagMeters}
            onChange={event => setDraft({...draft, maximumLagMeters: event.target.value})}
          />
        </label>
        <label htmlFor="kriging-fit-method">Variogram fitting method</label>
        <select
          id="kriging-fit-method"
          value={fitMethod}
          onChange={event => {
            setFitMethod(event.target.value as typeof fitMethod);
            setFit(null);
          }}
        >
          <option value="weighted-least-squares">Cressie weighted least squares</option>
          <option value="vecchia-likelihood">
            GPU Vecchia log-likelihood (up to 2,048 samples)
          </option>
          <option value="vecchia-fisher">GPU Vecchia Fisher scoring (up to 2,048 samples)</option>
        </select>
        <button
          type="button"
          disabled={fitting || analysis.rowCount < 2}
          onClick={async () => {
            setFitting(true);
            try {
              const base = buildSettings();
              const lag = Number(draft.maximumLagMeters);
              const suggestion = await analysis.suggestKrigingVariogram(base, {
                fitMethod,
                ...(draft.maximumLagMeters.trim() && lag > 0 ? {maximumLagMeters: lag} : {})
              });
              setFit(suggestion);
              setDraft(current => ({
                ...current,
                model: suggestion.settings.variogram.model,
                nugget: String(suggestion.settings.variogram.nugget),
                sill: String(suggestion.settings.variogram.sill),
                rangeMeters: String(suggestion.settings.variogram.rangeMeters)
              }));
              setError('');
            } catch (failure) {
              setError(failure instanceof Error ? failure.message : String(failure));
            } finally {
              setFitting(false);
            }
          }}
        >
          Fit variogram automatically
        </button>
        {fit && (
          <>
            <p className="hint" role="status">
              Fitted {fit.fit.model}: nugget {fit.fit.nugget.toPrecision(4)}, partial sill{' '}
              {fit.fit.sill.toPrecision(4)}, range {fit.fit.rangeMeters.toFixed(0)} m from{' '}
              {fit.fit.pairCount.toLocaleString()}{' '}
              {fit.fit.method === 'weighted-least-squares'
                ? `pairs in ${fit.fit.binCount} lags`
                : 'conditioning links'}
              {fit.fit.method === 'weighted-least-squares' && !fit.exhaustive
                ? ' (deterministically thinned)'
                : ''}
              . Compared families:{' '}
              {fit.fit.candidates
                .map(candidate => `${candidate.model} ${candidate.criterion.toPrecision(3)}`)
                .join(', ')}
              . Apply to use it.
            </p>
            {fit.fit.diagnostics && (
              <table aria-label="Variogram fit diagnostics">
                <thead>
                  <tr>
                    <th>Diagnostic</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(fit.fit.diagnostics).map(([name, value]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{Number.isFinite(value) ? value.toPrecision(6) : 'Unavailable'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
        <label htmlFor="kriging-confidenceLevel">
          Prediction interval level (blank: none, for example 0.95)
          <input
            id="kriging-confidenceLevel"
            type="number"
            min={0}
            max={1}
            step="any"
            value={draft.confidenceLevel}
            onChange={event => setDraft({...draft, confidenceLevel: event.target.value})}
          />
        </label>
        <label htmlFor="kriging-crossValidation">
          <input
            id="kriging-crossValidation"
            type="checkbox"
            checked={draft.crossValidation}
            onChange={event => setDraft({...draft, crossValidation: event.target.checked})}
          />{' '}
          Leave-one-out cross-validation
        </label>
        <p className="hint">
          Variogram parameters are explicit and replayable: type them or fit them from the data
          (Cressie-weighted least squares over spherical, exponential, Gaussian and Matérn families,
          on a deterministic pair sample, or a GPU Vecchia likelihood fit). Vecchia uses exponential
          or Matérn 3/2 covariance, a fitted constant mean, eight earlier neighbors in source order
          and a local meter frame. It reports convergence and does not fit a temporal covariance
          here; use the Space-time Vecchia catalog operation with an explicit time property for
          that. Nugget and partial sill must be nonnegative with a positive sum; range and search
          radius must be positive. Intervals are the kriging prediction plus or minus a normal
          multiplier times the model standard deviation; they are conditional on the variogram,
          assume Gaussian errors and ignore fitting uncertainty. Cross-validation kriges each
          retained sample from its nearest other samples; a root mean square standardized error near
          1 and coverage near the level suggest an adequate model.
        </p>
        <p className="hint">
          Repeated positions are averaged within the retained neighborhood, but still occupy
          neighbor slots. The minimum counts distinct positions. An exact center hit uses the mean
          at that position and returns zero variance, overriding the minimum. Nugget represents
          microscale variation, not observation error.
        </p>
        <p className="hint">
          Selection changes contributing samples and may find replacement nearest neighbors. Missing
          and nonfinite inputs are excluded. Float32 precision limits small distances and can make
          nearly singular neighborhoods unavailable. Radius-expanded grids must stay away from the
          poles and antimeridian.
        </p>
        <p className="hint">
          The map displays predictions. The grid inspector and separate downloads expose predictions
          in response units and model-based variance in squared response units; the inspector also
          shows interval bounds when a level is declared. Variance is conditional on the model; it
          does not establish calibrated prediction error or account for fitting uncertainty.
        </p>
        <p className="hint">
          Up to {MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS.toLocaleString()} source rows,{' '}
          {MAXIMUM_KEPLER_KRIGING_CELLS.toLocaleString()} grid cells and{' '}
          {MAXIMUM_KEPLER_KRIGING_GRID_DIMENSION} cells per axis, subject to device memory; the
          neighbor search uses a spatial index scaled to the source count. Each solve retains at
          most {MAXIMUM_KEPLER_KRIGING_NEIGHBORS} sample slots. The minimum distinct-position count
          cannot exceed the requested sample slots.
        </p>
        <button
          type="submit"
          disabled={analysis.rowCount < 1 || analysis.rowCount > MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS}
        >
          Apply model and run kriging
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
