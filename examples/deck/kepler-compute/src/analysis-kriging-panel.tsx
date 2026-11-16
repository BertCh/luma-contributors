// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useRef, useState} from 'react';
import {useAnalysisFormInitialization} from './analysis-form-initialization';
import {ResidentAnalysisSheetForm} from './analysis-sheet';
import {
  usePointAnalysisSheet,
  type PointSheetValues,
  type PointSheetSpecification
} from './analysis-point-weights-panel';
import {
  makeInterpolationSheetDefaults,
  createInterpolationSheetFields,
  getInterpolationSheetChecks,
  readInterpolationSheetNumber
} from './interpolation-panel';

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

export function makeKrigingSheetDefaults(
  settings: KeplerKrigingSettings | undefined,
  initialization: ReturnType<typeof useAnalysisFormInitialization>
) {
  return {
    ...makeInterpolationSheetDefaults(settings, initialization),
    searchRadiusMeters: settings?.searchRadiusMeters ?? 10000,
    neighborCount: settings?.neighborCount ?? 8,
    minimumNeighbors: settings?.minimumNeighbors ?? 3,
    model: settings?.variogram.model ?? 'spherical',
    nugget: settings?.variogram.nugget ?? 0,
    sill: settings?.variogram.sill ?? 1,
    rangeMeters: settings?.variogram.rangeMeters ?? 10000,
    confidenceLevel:
      settings?.confidenceLevel === undefined ? '' : String(settings.confidenceLevel),
    crossValidation: settings?.crossValidation ?? false,
    maximumLagMeters: '',
    fitMethod: settings?.variogramFit?.method ?? 'weighted-least-squares',
    editedFields: '[]'
  };
}

/** Preserve a replayed fit only while its coordinate fields and explicit variogram still match. */
export function getKrigingSheetSettings(
  values: PointSheetValues,
  fitted?: KeplerKrigingSettings
): KeplerKrigingSettings {
  const number = (name: string) => readInterpolationSheetNumber(values, name);
  const variogram = {
    model: values['model'],
    nugget: number('nugget'),
    sill: number('sill'),
    rangeMeters: number('rangeMeters')
  };
  const fitMatches =
    fitted &&
    fitted.longitudeColumn === values['longitudeColumn'] &&
    fitted.latitudeColumn === values['latitudeColumn'] &&
    fitted.variogram.model === variogram.model &&
    // The recipe parser normalizes model numbers to Float32; compare in that same domain.
    fitted.variogram.nugget === Math.fround(variogram.nugget) &&
    fitted.variogram.sill === Math.fround(variogram.sill) &&
    fitted.variogram.rangeMeters === Math.fround(variogram.rangeMeters);
  return parseKeplerKrigingSettings({
    longitudeColumn: values['longitudeColumn'],
    latitudeColumn: values['latitudeColumn'],
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters',
    grid: {
      bounds: ['west', 'south', 'east', 'north'].map(number),
      width: number('width'),
      height: number('height')
    },
    searchRadiusMeters: number('searchRadiusMeters'),
    neighborCount: number('neighborCount'),
    minimumNeighbors: number('minimumNeighbors'),
    variogram,
    ...(String(values['confidenceLevel'] ?? '').trim()
      ? {confidenceLevel: number('confidenceLevel')}
      : {}),
    ...(values['crossValidation'] ? {crossValidation: true} : {}),
    ...(fitMatches && fitted.variogramFit ? {variogramFit: fitted.variogramFit} : {})
  });
}

/** The generated sheet retains resident fitting, intervals, cross-validation and result views. */
export function KrigingPanel({analysis}: {analysis: AnalysisRuntime}) {
  const settings = analysis.settings.kriging;
  const responseColumn = analysis.settings.column;
  const initialization = useAnalysisFormInitialization(analysis);
  const scope = analysis.coordinateRestorationKey ?? analysis.datasetId;
  const sheet = usePointAnalysisSheet(
    makeKrigingSheetDefaults(settings, initialization),
    settings,
    scope,
    initialization.coordinateSource
  );
  const [error, setError] = useState('');
  const [fit, setFit] = useState<(KrigingVariogramSuggestion & {responseColumn: string}) | null>(
    null
  );
  const [fitting, setFitting] = useState(false);
  const fitRequest = useRef(0);
  const savedSignature = JSON.stringify(settings);
  const signature = JSON.stringify([scope, savedSignature, responseColumn, sheet.values]);
  const fitScope = useRef(signature);
  // Editing or restoring during an asynchronous fit makes that fit obsolete before it can publish.
  if (fitScope.current !== signature) {
    fitScope.current = signature;
    fitRequest.current++;
  }
  useEffect(() => {
    setFit(null);
    setError('');
    setFitting(false);
    return () => {
      fitRequest.current++;
    };
  }, [scope, savedSignature, responseColumn]);
  useEffect(() => {
    setFitting(false);
  }, [signature]);
  const buildSettings = (values = sheet.values) =>
    getKrigingSheetSettings(
      values,
      fit?.responseColumn === analysis.settings.column ? fit.settings : analysis.settings.kriging
    );
  const specifications: PointSheetSpecification[] = [
    {
      name: 'model',
      label: 'Kriging variogram model',
      controlId: 'kriging-model',
      kind: 'enum',
      options: MODELS
    },
    ...NUMERIC_FIELDS.map(field => ({
      name: field.name,
      label: field.label,
      controlId: `kriging-${field.name}`,
      kind: field.step === '1' ? ('integer' as const) : ('number' as const),
      minimum: field.minimum,
      maximum: 'maximum' in field ? field.maximum : undefined,
      level: ['neighborCount', 'minimumNeighbors'].includes(field.name)
        ? ('advanced' as const)
        : ('basic' as const)
    })),
    {
      name: 'maximumLagMeters',
      label: 'Variogram maximum lag in meters (blank: a third of the data extent)',
      controlId: 'kriging-maximumLagMeters',
      kind: 'string',
      maximumLength: 64,
      level: 'advanced'
    },
    {
      name: 'fitMethod',
      label: 'Variogram fitting method',
      controlId: 'kriging-fit-method',
      kind: 'enum',
      options: ['weighted-least-squares', 'vecchia-likelihood', 'vecchia-fisher'],
      level: 'advanced',
      description:
        'Cressie weighted least squares, GPU Vecchia log-likelihood, or GPU Vecchia Fisher scoring. Vecchia fits support up to 2,048 samples.'
    },
    {
      name: 'confidenceLevel',
      label: 'Prediction interval level (blank: none, for example 0.95)',
      controlId: 'kriging-confidenceLevel',
      kind: 'string',
      maximumLength: 64,
      level: 'advanced'
    },
    {
      name: 'crossValidation',
      label: 'Leave-one-out cross-validation',
      controlId: 'kriging-crossValidation',
      kind: 'boolean',
      level: 'advanced'
    }
  ];
  const fields = createInterpolationSheetFields(
    sheet.values,
    analysis.availableColumns,
    'kriging',
    specifications,
    name => {
      if (
        fit?.responseColumn === responseColumn &&
        name in fit.settings.variogram &&
        sheet.values[name] ===
          fit.settings.variogram[name as keyof KeplerKrigingSettings['variogram']]
      )
        return `Fitted variogram (${fit.settings.variogramFit?.method})`;
      return !settings &&
        ['west', 'south', 'east', 'north'].includes(name) &&
        sheet.source(name) !== 'User override'
        ? initialization.boundsSource
        : sheet.source(name);
    }
  );
  const checks = getInterpolationSheetChecks(
    fields,
    analysis.rowCount,
    MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS,
    () => buildSettings()
  );
  const maximumLag = String(sheet.values['maximumLagMeters'] ?? '').trim();
  const lagProblem =
    maximumLag && (!Number.isFinite(Number(maximumLag)) || Number(maximumLag) <= 0)
      ? 'Variogram maximum lag must be positive meters or blank for a third of the data extent.'
      : '';
  return (
    <details className="analysis-kriging-settings">
      <summary>Configure ordinary kriging grid</summary>
      <ResidentAnalysisSheetForm
        key={JSON.stringify([scope, settings])}
        fields={fields}
        checks={checks}
        method="Resident ordinary kriging predicts the selected numeric response on an explicit north-up geographic grid, using spherical great-circle meters. It offers fitted variograms, intervals and leave-one-out cross-validation. Catalog kriging uses a supplied variogram without fitting or cross-validation; the curated kriging analysis includes separate diagnostics."
        submitLabel="Apply model and run kriging"
        onDraftChange={() => {
          fitRequest.current++;
          setFitting(false);
        }}
        onChange={(name, value) => {
          fitRequest.current++;
          setFitting(false);
          sheet.change(name, value);
        }}
        onApply={current => analysis.configureKriging(buildSettings({...sheet.values, ...current}))}
        actions={[
          {
            label: 'Fit variogram automatically',
            disabled: fitting || analysis.rowCount < 2 || Boolean(lagProblem),
            onInvoke: async current => {
              const request = ++fitRequest.current;
              setFitting(true);
              try {
                const values = {...sheet.values, ...current};
                const base = buildSettings(values);
                const lagText = String(values['maximumLagMeters'] ?? '').trim();
                const lag = Number(lagText);
                if (lagText && (!Number.isFinite(lag) || lag <= 0))
                  throw new Error('Variogram maximum lag must be positive meters or blank');
                const suggestion = await analysis.suggestKrigingVariogram(base, {
                  fitMethod: values['fitMethod'] as NonNullable<
                    KeplerKrigingSettings['variogramFit']
                  >['method'],
                  ...(lagText ? {maximumLagMeters: lag} : {})
                });
                if (request !== fitRequest.current || analysis.settings.column !== responseColumn)
                  return;
                const fittedSettings = parseKeplerKrigingSettings(suggestion.settings);
                setFit({...suggestion, settings: fittedSettings, responseColumn});
                sheet.change('model', fittedSettings.variogram.model, {
                  nugget: fittedSettings.variogram.nugget,
                  sill: fittedSettings.variogram.sill,
                  rangeMeters: fittedSettings.variogram.rangeMeters
                });
                setError('');
              } catch (failure) {
                if (request === fitRequest.current)
                  setError(failure instanceof Error ? failure.message : String(failure));
              } finally {
                if (request === fitRequest.current) setFitting(false);
              }
            }
          }
        ]}
      />
      <section aria-label="Variogram fitting">
        <p className="hint">
          Fit the current draft, then Apply to run the model. Fitting does not replace the accepted
          grid.
        </p>
        {lagProblem && <p role="alert">{lagProblem}</p>}
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
      </section>
      <p className="hint">
        Variogram parameters are explicit and replayable: type them or fit them from the data
        (Cressie-weighted least squares over spherical, exponential, Gaussian and Matérn families,
        on a deterministic pair sample, or a GPU Vecchia likelihood fit). Vecchia uses exponential
        or Matérn 3/2 covariance, a fitted constant mean, eight earlier neighbors in source order
        and a local meter frame. It reports convergence and does not fit a temporal covariance here;
        use the Space-time Vecchia catalog operation with an explicit time property for that. Nugget
        and partial sill must be nonnegative with a positive sum; range and search radius must be
        positive. Intervals are the kriging prediction plus or minus a normal multiplier times the
        model standard deviation; they are conditional on the variogram, assume Gaussian errors and
        ignore fitting uncertainty. Cross-validation kriges each retained sample from its nearest
        other samples; a root mean square standardized error near 1 and coverage near the level
        suggest an adequate model.
      </p>
      <p className="hint">
        Repeated positions are averaged within the retained neighborhood, but still occupy neighbor
        slots. The minimum counts distinct positions. An exact center hit uses the mean at that
        position and returns zero variance, overriding the minimum. Nugget represents microscale
        variation, not observation error.
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
        neighbor search uses a spatial index scaled to the source count. Each solve retains at most{' '}
        {MAXIMUM_KEPLER_KRIGING_NEIGHBORS} sample slots. The minimum distinct-position count cannot
        exceed the requested sample slots.
      </p>
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
