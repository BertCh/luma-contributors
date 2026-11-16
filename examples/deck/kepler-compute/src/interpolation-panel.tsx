// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {useAnalysisFormInitialization} from './analysis-form-initialization';

import {
  MAXIMUM_KEPLER_IDW_SOURCE_ROWS,
  parseKeplerIDWSettings,
  type KeplerIDWSettings,
  type KeplerKrigingSettings,
  type CatalogParameterValue
} from '@luma-contributors/kepler';
import {ResidentAnalysisSheetForm, type ResidentAnalysisSheetField} from './analysis-sheet';
import {
  usePointAnalysisSheet,
  makePointSheetColumn,
  type PointSheetValues,
  type PointSheetSpecification
} from './analysis-point-weights-panel';
import type {AnalysisRuntime} from './analysis-runtime';
import {interpolationSurfaceHover} from './interpolation-surface-hover';
import type {InterpolationCell} from './interpolation-runtime';

const FIELDS = [
  {name: 'west', label: 'Interpolation west longitude', minimum: -180, maximum: 180, step: 'any'},
  {name: 'south', label: 'Interpolation south latitude', minimum: -90, maximum: 90, step: 'any'},
  {name: 'east', label: 'Interpolation east longitude', minimum: -180, maximum: 180, step: 'any'},
  {name: 'north', label: 'Interpolation north latitude', minimum: -90, maximum: 90, step: 'any'},
  {name: 'width', label: 'Interpolation columns', minimum: 1, maximum: 8192, step: '1'},
  {name: 'height', label: 'Interpolation rows', minimum: 1, maximum: 8192, step: '1'},
  {name: 'searchRadius', label: 'Interpolation search radius in meters', minimum: 1, step: 'any'},
  {name: 'power', label: 'Interpolation distance power', minimum: 0, maximum: 32, step: 'any'},
  {
    name: 'neighborCount',
    label: 'Interpolation nearest samples',
    minimum: 1,
    maximum: 64,
    step: '1'
  },
  {
    name: 'minimumNeighborCount',
    label: 'Interpolation minimum samples',
    minimum: 1,
    maximum: 64,
    step: '1'
  }
] as const;

/** Shared resident grid defaults retain AX publication roles and data-derived geographic bounds. */
export function makeInterpolationSheetDefaults(
  settings: KeplerIDWSettings | KeplerKrigingSettings | undefined,
  initialization: ReturnType<typeof useAnalysisFormInitialization>
): Record<string, CatalogParameterValue> {
  return {
    ...initialization.interpolation,
    ...(settings
      ? {
          longitudeColumn: settings.longitudeColumn,
          latitudeColumn: settings.latitudeColumn,
          west: settings.grid.bounds[0],
          south: settings.grid.bounds[1],
          east: settings.grid.bounds[2],
          north: settings.grid.bounds[3]
        }
      : {}),
    width: settings?.grid.width ?? 32,
    height: settings?.grid.height ?? 32
  };
}

/** Both resident interpolators use the same sheet controls, source explanations and role checks. */
export function createInterpolationSheetFields(
  values: PointSheetValues,
  columns: readonly string[],
  prefix: string,
  specifications: readonly PointSheetSpecification[],
  source: (name: string) => string
): ResidentAnalysisSheetField[] {
  const label = prefix === 'kriging' ? 'Kriging' : 'Interpolation';
  return [
    makePointSheetColumn(
      'longitudeColumn',
      `${label} longitude field`,
      `${prefix}-longitudeColumn`
    ),
    makePointSheetColumn('latitudeColumn', `${label} latitude field`, `${prefix}-latitudeColumn`),
    ...specifications
  ].map(({controlId, ...specification}) => ({
    specification,
    controlId,
    value: values[specification.name] ?? '',
    source: source(specification.name),
    ...(specification.fieldRole ? {columns} : {})
  }));
}

export function getInterpolationSheetChecks(
  fields: readonly ResidentAnalysisSheetField[],
  rowCount: number,
  maximumRows: number,
  readSettings: () => unknown
): string[] {
  const failures = fields.flatMap(field =>
    field.specification.fieldRole && !field.columns?.includes(String(field.value))
      ? [`${field.specification.label}: choose an available prepared numeric column.`]
      : []
  );
  if (rowCount < 1 || rowCount > maximumRows)
    failures.push(
      `Interpolation source rows must be from 1 through ${maximumRows.toLocaleString()}; choose a supported dataset.`
    );
  try {
    readSettings();
  } catch (failure) {
    failures.push(
      `${failure instanceof Error ? failure.message : String(failure)}. Review the grid bounds, model and limits before applying.`
    );
  }
  return failures;
}

/** Blank numeric controls must not become zero through Number(''). */
export function readInterpolationSheetNumber(values: PointSheetValues, name: string): number {
  if (String(values[name] ?? '').trim() === '') throw new Error(`Declare interpolation ${name}`);
  return Number(values[name]);
}

export function getIDWSheetSettings(values: PointSheetValues): KeplerIDWSettings {
  const number = (name: string) => readInterpolationSheetNumber(values, name);
  return parseKeplerIDWSettings({
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
    searchRadius: number('searchRadius'),
    power: number('power'),
    neighborCount: number('neighborCount'),
    minimumNeighborCount: number('minimumNeighborCount'),
    taper: values['taper']
  });
}

/** Grid controls deliberately identify a new cell space, independent of source rows. */
export function InterpolationPanel({analysis}: {analysis: AnalysisRuntime}) {
  const settings = analysis.settings.idw;
  const initialization = useAnalysisFormInitialization(analysis);
  const sheet = usePointAnalysisSheet(
    {
      ...makeInterpolationSheetDefaults(settings, initialization),
      searchRadius: settings?.searchRadius ?? 10000,
      power: settings?.power ?? 2,
      neighborCount: settings?.neighborCount ?? 8,
      minimumNeighborCount: settings?.minimumNeighborCount ?? 3,
      taper: settings?.taper ?? 'none',
      editedFields: '[]'
    },
    settings,
    analysis.coordinateRestorationKey ?? analysis.datasetId,
    initialization.coordinateSource
  );
  const fields = createInterpolationSheetFields(
    sheet.values,
    analysis.availableColumns,
    'interpolation',
    [
      ...FIELDS.map(field => ({
        name: field.name,
        label: field.label,
        controlId: `interpolation-${field.name}`,
        kind: field.step === '1' ? ('integer' as const) : ('number' as const),
        minimum: field.minimum,
        maximum: 'maximum' in field ? field.maximum : undefined,
        level: ['neighborCount', 'minimumNeighborCount'].includes(field.name)
          ? ('advanced' as const)
          : ('basic' as const)
      })),
      {
        name: 'taper',
        label: 'Interpolation radius taper',
        controlId: 'interpolation-taper',
        kind: 'enum',
        options: ['none', 'modified-shepard'],
        level: 'advanced'
      }
    ],
    name =>
      !settings &&
      ['west', 'south', 'east', 'north'].includes(name) &&
      sheet.source(name) !== 'User override'
        ? initialization.boundsSource
        : sheet.source(name)
  );
  return (
    <details className="analysis-interpolation-settings">
      <summary>Configure inverse-distance interpolation grid</summary>
      <ResidentAnalysisSheetForm
        key={JSON.stringify([analysis.coordinateRestorationKey, settings])}
        fields={fields}
        checks={getInterpolationSheetChecks(
          fields,
          analysis.rowCount,
          MAXIMUM_KEPLER_IDW_SOURCE_ROWS,
          () => getIDWSheetSettings(sheet.values)
        )}
        method="Resident IDW predicts the selected numeric field on a separate north-up longitude/latitude grid using spherical great-circle meters. It offers hard-radius inverse distance and Modified Shepard tapering. Catalog IDW uses its own cell-size grid controls; this resident method retains explicit bounds, dimensions and source-row tie breaking."
        submitLabel="Apply grid and run interpolation"
        onChange={(name, value) => sheet.change(name, value)}
        onApply={current =>
          analysis.configureInterpolation(getIDWSheetSettings({...sheet.values, ...current}))
        }
      />
      <p className="hint">
        Up to 16.7 million source rows and cells (8,192 cells per axis); large grids run as tiled
        GPU dispatches and are limited by device memory. Radius-expanded grids must stay away from
        the poles and antimeridian. Float32 coordinates limit distance precision; source nulls and
        nonfinite values are excluded.
      </p>
      <p className="hint">
        Exact center hits use the smallest original source row and override the minimum-sample
        threshold. Otherwise sparse cells are transparent nodata. Ties among nearest samples use
        source-row order. Selection changes the contributing samples and can find replacement
        nearest neighbors. This grid does not use the fixed statistical neighbor cache.
      </p>
      <p className="hint">
        Power 0 gives an unweighted mean. This method supplies values and contributing counts,
        without kriging variance, fitted parameters or cross-validation claims.
      </p>
    </details>
  );
}

export function InterpolationResults({analysis}: {analysis: AnalysisRuntime}) {
  const summary = analysis.summary;
  const grid = summary?.interpolation;
  const [offset, setOffset] = useState(0);
  const [rows, setRows] = useState<readonly InterpolationCell[]>([]);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [exporting, setExporting] = useState<'prediction' | 'variance' | null>(null);
  const isKriging = summary?.operation === 'kriging-points';
  const kriging = summary?.kriging;
  const revision = summary
    ? `${summary.sourceRevision}:${summary.revision}:${summary.selectionRevision}`
    : '';
  useEffect(() => {
    setOffset(0);
  }, [revision]);
  useEffect(() => {
    let active = true;
    if (!grid) {
      setRows([]);
      return;
    }
    setPending(true);
    setRows([]);
    setError('');
    void analysis
      .readInterpolationPage(Math.min(offset, grid.cellCount - 1), 16)
      .then(values => {
        if (active) setRows(values);
      })
      .catch(failure => {
        if (active) {
          setRows([]);
          setError(failure instanceof Error ? failure.message : String(failure));
        }
      })
      .finally(() => {
        if (active) setPending(false);
      });
    return () => {
      active = false;
    };
  }, [analysis, revision, offset, grid?.cellCount]);
  if (!grid) return null;
  const hasInterval = rows.some(
    row => row.operation === 'kriging-points' && row.lower !== undefined
  );
  const download = (metric: 'prediction' | 'variance') => {
    setExporting(metric);
    setError('');
    void analysis
      .exportInterpolationGeoTIFF(metric)
      .then(result => {
        try {
          if (!result.isCurrent()) throw new Error('Interpolation grid changed before download');
          const url = URL.createObjectURL(new Blob([result.data], {type: 'image/tiff'}));
          const anchor = document.createElement('a');
          try {
            anchor.href = url;
            anchor.download = result.filename;
            document.body.append(anchor);
            anchor.click();
          } finally {
            anchor.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
          }
        } finally {
          result.destroy();
        }
      })
      .catch(failure => setError(failure instanceof Error ? failure.message : String(failure)))
      .finally(() => setExporting(null));
  };
  return (
    <details className="interpolation-grid-results">
      <summary>Inspect interpolation grid cells</summary>
      <p className="hint">
        {grid.width} × {grid.height} new cells · {grid.validSourceCount} contributing source
        candidates. Row 0 is the north edge. Grid values remain on the GPU; this page reads at most
        16 cells.
      </p>
      <label>
        <input
          type="checkbox"
          defaultChecked={interpolationSurfaceHover.isInspectionEnabled}
          onChange={event => analysis.setSurfaceInspection(event.currentTarget.checked)}
          aria-describedby="interpolation-surface-hint"
        />{' '}
        Show interpolated value when hovering the surface
      </label>
      <p id="interpolation-surface-hint" className="hint">
        While on, the surface captures the pointer, so points beneath it are not hoverable. Escape
        dismisses the popover.
      </p>
      {isKriging && (
        <section aria-label="Kriging grid diagnostics">
          <h3>Prediction and variance diagnostics</h3>
          {kriging ? (
            <>
              <p>
                {kriging.invalidCount.toLocaleString()} cells unavailable from invalid settings or
                insufficient distinct neighbors · {kriging.uncertainCount.toLocaleString()} cells
                unavailable from numerically unreliable solutions
              </p>
              <p>
                Variance: {formatGridValue(kriging.variance.minimum)} to{' '}
                {formatGridValue(kriging.variance.maximum)} squared response units · mean{' '}
                {formatGridValue(kriging.variance.mean)} · {kriging.variance.count.toLocaleString()}{' '}
                defined and {kriging.variance.nullCount.toLocaleString()} nodata cells
              </p>
              {kriging.interval && (
                <p>
                  {(kriging.interval.level * 100).toFixed(1)}% prediction interval half-width (z ={' '}
                  {kriging.interval.multiplier.toFixed(3)}):{' '}
                  {formatGridValue(kriging.interval.halfWidthAtMinimumVariance)} at the lowest
                  variance, {formatGridValue(kriging.interval.halfWidthAtMeanVariance)} at the mean
                  variance, {formatGridValue(kriging.interval.halfWidthAtMaximumVariance)} at the
                  highest
                </p>
              )}
              {kriging.crossValidation && (
                <p>
                  Leave-one-out cross-validation over{' '}
                  {kriging.crossValidation.count.toLocaleString()} retained samples: mean error{' '}
                  {formatGridValue(kriging.crossValidation.meanError)} · RMSE{' '}
                  {formatGridValue(kriging.crossValidation.rootMeanSquareError)} · mean standardized
                  error {formatGridValue(kriging.crossValidation.meanStandardizedError)} · RMS
                  standardized error{' '}
                  {formatGridValue(kriging.crossValidation.rootMeanSquareStandardizedError)} (near 1
                  suggests calibrated variance) · interval coverage{' '}
                  {formatGridValue(kriging.crossValidation.coverage)} at level{' '}
                  {kriging.crossValidation.level}
                </p>
              )}
            </>
          ) : (
            <p role="status">Kriging diagnostics are unavailable for this grid.</p>
          )}
          <p className="hint">
            The map displays predictions. Variance is model-based uncertainty in squared response
            units, conditional on the supplied variogram. A declared prediction interval is the
            prediction plus or minus a normal multiplier times the model standard deviation; it
            ignores variogram estimation uncertainty. Leave-one-out diagnostics, when requested,
            appear above and do not change the map. An exact hit has zero variance under this model.
          </p>
          <p className="hint">
            Unavailable predictions and variances remain nodata. Diagnostic counts describe the
            whole grid; a specific cause and retained sample count are not available for each cell.
          </p>
        </section>
      )}
      <div style={{overflowX: 'auto'}}>
        <table aria-label="Interpolation grid cells">
          <thead>
            <tr>
              <th scope="col">Cell</th>
              <th scope="col">Longitude</th>
              <th scope="col">Latitude</th>
              <th scope="col">{isKriging ? 'Prediction' : 'Value'}</th>
              <th scope="col">{isKriging ? 'Variance (squared response units)' : 'Samples'}</th>
              {hasInterval && <th scope="col">Interval lower</th>}
              {hasInterval && <th scope="col">Interval upper</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={row.cellIndex}>
                <td>{row.cellIndex}</td>
                <td>{row.longitude.toFixed(5)}</td>
                <td>{row.latitude.toFixed(5)}</td>
                <td>{formatGridValue(row.value)}</td>
                <td>
                  {row.operation === 'kriging-points'
                    ? formatGridValue(row.variance)
                    : row.neighbors}
                </td>
                {hasInterval && (
                  <td>
                    {row.operation === 'kriging-points' ? formatGridValue(row.lower ?? null) : ''}
                  </td>
                )}
                {hasInterval && (
                  <td>
                    {row.operation === 'kriging-points' ? formatGridValue(row.upper ?? null) : ''}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">
        Each download reads one complete bounded Float32 grid band in EPSG:4326. Nodata stays
        missing; the file includes the recipe and source revision.{' '}
        {isKriging
          ? 'Prediction and variance are separate downloads with explicit band provenance; variance uses squared response units.'
          : 'Sample counts are available in the inspector.'}
      </p>
      {isKriging ? (
        <div className="actions">
          <button
            type="button"
            disabled={exporting !== null}
            onClick={() => download('prediction')}
          >
            {exporting === 'prediction'
              ? 'Preparing prediction download…'
              : 'Download kriging prediction GeoTIFF'}
          </button>
          <button type="button" disabled={exporting !== null} onClick={() => download('variance')}>
            {exporting === 'variance'
              ? 'Preparing variance download…'
              : 'Download kriging variance GeoTIFF'}
          </button>
        </div>
      ) : (
        <button type="button" disabled={exporting !== null} onClick={() => download('prediction')}>
          {exporting ? 'Preparing grid download…' : 'Download grid GeoTIFF'}
        </button>
      )}
      <div className="actions">
        <button
          type="button"
          disabled={pending || offset === 0}
          onClick={() => setOffset(Math.max(0, offset - 16))}
        >
          Previous grid cells
        </button>
        <button
          type="button"
          disabled={pending || offset + 16 >= grid.cellCount}
          onClick={() => setOffset(offset + 16)}
        >
          Next grid cells
        </button>
      </div>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
    </details>
  );
}

function formatGridValue(value: number | null): string {
  return value !== null && Number.isFinite(value)
    ? value.toLocaleString(undefined, {maximumSignificantDigits: 6})
    : 'nodata';
}
