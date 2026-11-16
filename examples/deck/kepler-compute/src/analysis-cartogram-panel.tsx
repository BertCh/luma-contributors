// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MAXIMUM_KEPLER_DORLING_SOURCE_ROWS,
  MAXIMUM_KEPLER_DORLING_ITERATIONS,
  parseKeplerDorlingSettings,
  type KeplerDorlingSettings,
  type CatalogParameterValue
} from '@luma-contributors/kepler';
import {useAnalysisFormInitialization} from './analysis-form-initialization';
import type {AnalysisRuntime} from './analysis-runtime';
import {CartogramDeclarationForm} from './analysis2-cartogram-declaration-form';
import {ResidentConsumerNotice} from './resident-consumer-notice';
import {ResidentAnalysisSheetForm} from './analysis-sheet';
import {
  createResidentSheetFields,
  getResidentSheetChecks,
  readResidentSheetNumber
} from './analysis-sheet-model';
import {
  usePointAnalysisSheet,
  makePointSheetColumn,
  type PointSheetValues,
  type PointSheetSpecification
} from './analysis-point-weights-panel';

export function makeCartogramSheetDefaults(
  settings: KeplerDorlingSettings | undefined,
  valueColumn: string,
  initialization: ReturnType<typeof useAnalysisFormInitialization>
): Record<string, CatalogParameterValue> {
  return {
    xColumn: settings?.xColumn ?? initialization.coordinates.xColumn,
    yColumn: settings?.yColumn ?? initialization.coordinates.yColumn,
    valueColumn: settings?.valueColumn ?? valueColumn,
    crs: settings?.crs ?? '',
    minX: settings?.bounds[0] ?? initialization.coordinates.minX,
    minY: settings?.bounds[1] ?? initialization.coordinates.minY,
    maxX: settings?.bounds[2] ?? initialization.coordinates.maxX,
    maxY: settings?.bounds[3] ?? initialization.coordinates.maxY,
    areaScale: settings?.areaScale ?? 1,
    maximumIterations: settings?.maximumIterations ?? 100,
    overlapTolerance: settings?.overlapTolerance ?? 0.001,
    attraction: settings?.attraction ?? 0.01,
    editedFields: '[]'
  };
}

export function getCartogramSheetSettings(values: PointSheetValues): KeplerDorlingSettings {
  const number = (name: string) => readResidentSheetNumber(values, name);
  return parseKeplerDorlingSettings({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    xColumn: values['xColumn'],
    yColumn: values['yColumn'],
    valueColumn: values['valueColumn'],
    crs: values['crs'],
    bounds: ['minX', 'minY', 'maxX', 'maxY'].map(number),
    areaScale: number('areaScale'),
    maximumIterations: number('maximumIterations'),
    overlapTolerance: number('overlapTolerance'),
    attraction: number('attraction')
  });
}

/** Saved cartograms own their value and coordinate roles independently of the general numeric field. */
export function CartogramPanel({analysis}: {analysis: AnalysisRuntime}) {
  const initialization = useAnalysisFormInitialization(analysis);
  const saved = analysis.settings.cartogram;
  const sheet = usePointAnalysisSheet(
    makeCartogramSheetDefaults(saved, analysis.settings.column, initialization),
    saved,
    analysis.coordinateRestorationKey ?? analysis.datasetId,
    initialization.coordinateSource
  );
  const specifications: PointSheetSpecification[] = [
    makePointSheetColumn('xColumn', 'Cartogram projected X field', 'analysis-cartogram-xColumn'),
    makePointSheetColumn('yColumn', 'Cartogram projected Y field', 'analysis-cartogram-yColumn'),
    makePointSheetColumn(
      'valueColumn',
      'Cartogram nonnegative area-value field',
      'analysis-cartogram-valueColumn'
    ),
    {
      name: 'crs',
      label: 'Cartogram declared projected CRS',
      controlId: 'analysis-cartogram-crs',
      kind: 'string',
      maximumLength: 1024
    },
    ...(['minX', 'minY', 'maxX', 'maxY'] as const).map((name, index) => ({
      name,
      label: `Cartogram ${['minimum X', 'minimum Y', 'maximum X', 'maximum Y'][index]} in projected meters`,
      controlId: `analysis-cartogram-${name}`,
      kind: 'number' as const
    })),
    {
      name: 'areaScale',
      label: 'Cartogram area scale in projected square meters per value',
      controlId: 'analysis-cartogram-areaScale',
      kind: 'number',
      minimum: 0
    },
    {
      name: 'maximumIterations',
      label: 'Cartogram maximum iterations',
      controlId: 'analysis-cartogram-maximumIterations',
      kind: 'integer',
      minimum: 0,
      maximum: MAXIMUM_KEPLER_DORLING_ITERATIONS,
      level: 'advanced'
    },
    {
      name: 'overlapTolerance',
      label: 'Cartogram overlap tolerance in projected meters',
      controlId: 'analysis-cartogram-overlapTolerance',
      kind: 'number',
      minimum: 0,
      level: 'advanced'
    },
    {
      name: 'attraction',
      label: 'Cartogram attraction',
      controlId: 'analysis-cartogram-attraction',
      kind: 'number',
      minimum: 0,
      maximum: 1,
      level: 'advanced'
    }
  ];
  const fields = createResidentSheetFields(
    sheet.values,
    specifications,
    name => {
      const source = sheet.source(name);
      if (saved || source === 'User override') return source;
      if (['minX', 'minY', 'maxX', 'maxY'].includes(name))
        return 'Dataset extent in source coordinate units; confirm projected meter bounds';
      if (name === 'crs') return 'Explicit projected CRS declaration required';
      if (name === 'valueColumn') return 'Numeric field above (publication role or user selection)';
      return source;
    },
    () => analysis.availableColumns
  );
  const checks = getResidentSheetChecks(fields, () => getCartogramSheetSettings(sheet.values));
  if (analysis.rowCount < 1 || analysis.rowCount > MAXIMUM_KEPLER_DORLING_SOURCE_ROWS)
    checks.push(`Cartograms support 1 to ${MAXIMUM_KEPLER_DORLING_SOURCE_ROWS} source rows.`);
  return (
    <details className="analysis-cartogram-panel">
      <summary>Configure Dorling circle cartogram</summary>
      <ResidentAnalysisSheetForm
        key={JSON.stringify([analysis.coordinateRestorationKey, saved])}
        fields={fields}
        checks={checks}
        method="Resident Dorling cartogram: X and Y must already be projected meter coordinates. Declare their CRS and finite bounds; declaring a CRS does not project or verify coordinates. Each eligible row becomes a circle with area = nonnegative value × area scale in projected square meters, not ground area. The current analysis population selects source rows. Zero-area circles remain stationary. Missing, nonfinite and negative values are excluded while retaining source identity. Layout is heuristic; zero iterations returns the initial layout and convergence is not guaranteed. Finite exhausted layouts remain available with diagnostics. Float32 precision can limit movement; coordinates are not recentered. Arbitrary declared projected CRSs produce table results; a map overlay requires EPSG:3857 and a compatible map projection contract."
        submitLabel="Apply fields and run cartogram"
        onDraftChange={sheet.markEdited}
        onChange={sheet.change}
        onApply={current =>
          analysis.configureCartogram(getCartogramSheetSettings({...sheet.values, ...current}))
        }
      />
      {saved && saved.crs !== 'EPSG:3857' && <CartogramDeclarationForm analysis={analysis} />}
    </details>
  );
}

/** Layout exhaustion is reported as a diagnostic while finite source-aligned results remain usable. */
export function CartogramResults({analysis}: {analysis: AnalysisRuntime}) {
  if (analysis.settings.mode !== 'dorling-cartogram') return null;
  const summary = analysis.cartogramSummary;
  const unavailableReason = analysis.cartogramUnavailableReason;
  if (!summary) {
    return (
      <section aria-label="Cartogram results">
        {analysis.error ? (
          <ResidentConsumerNotice
            consumer="analysis"
            status="error"
            className="error"
            reason={analysis.error}
          />
        ) : (
          <p role="status">Computing cartogram layout and overlap diagnostics…</p>
        )}
        {unavailableReason && <p className="hint">Map overlay unavailable: {unavailableReason}</p>}
      </section>
    );
  }
  const diagnostics = summary.cartogram;
  const status = diagnostics.converged
    ? 'Converged within overlap tolerance'
    : diagnostics.budgetExhausted
      ? 'Iteration budget exhausted; finite layout available'
      : 'Finite layout available; overlap tolerance not reached';
  return (
    <section aria-label="Cartogram results">
      <h3>Dorling circle cartogram</h3>
      <p role="status">{status}</p>
      <p>
        {summary.count.toLocaleString()} eligible circles ·{' '}
        {diagnostics.excludedCount.toLocaleString()} excluded or unselected source rows ·{' '}
        {diagnostics.iterations.toLocaleString()} iterations
      </p>
      <dl>
        <dt>Maximum remaining overlap penetration</dt>
        <dd>{formatMetric(diagnostics.maximumPenetration)} projected meters</dd>
        <dt>Maximum displacement from original centers</dt>
        <dd>{formatMetric(diagnostics.maximumDisplacement)} projected meters</dd>
        <dt>Circle radius range</dt>
        <dd>
          {formatMetric(summary.minimum)} to {formatMetric(summary.maximum)} projected meters
        </dd>
        <dt>Mean circle radius</dt>
        <dd>{formatMetric(summary.mean)} projected meters</dd>
        <dt>Estimated Float32 coordinate spacing at the declared extent</dt>
        <dd>{formatMetric(diagnostics.coordinateResolutionMeters)} projected meters</dd>
        <dt>Declared projected CRS</dt>
        <dd>{diagnostics.crs}</dd>
      </dl>
      {diagnostics.precisionWarning && (
        <p className="hint">Float32 precision: {diagnostics.precisionWarning}.</p>
      )}
      <p className="hint">
        The source table retains each row's layout center, radius and source identity. Excluded rows
        include invalid coordinates or values and rows outside the selected population. A zero-area
        row remains a stationary zero-radius circle. Original source geometry is retained.
      </p>
      {diagnostics.budgetExhausted && !diagnostics.converged && (
        <p className="hint">
          Remaining overlap does not invalidate a finite layout. The iteration limit bounds work; it
          does not guarantee separation. Zero iterations return the initial layout. Review the
          residual overlap before interpreting circle placement.
        </p>
      )}
      <p className="hint">
        {unavailableReason
          ? 'This result is available in the source table.'
          : diagnostics.display === 'web-mercator-overlay'
            ? 'This result supports an EPSG:3857 Web Mercator overlay.'
            : 'This result is drawn on a declared local tangent plane (display only).'}{' '}
        Overlay display requires EPSG:3857 coordinates and a compatible native Web Mercator map
        view. Projected meter radii and area do not represent ground distance or ground area. The
        declared CRS does not automatically project coordinates or replace source geometry.
      </p>
      {unavailableReason && <p className="hint">Map overlay unavailable: {unavailableReason}</p>}
      <p className="hint">
        This is a heuristic layout, with no GeoDa coordinate parity claim. Float32 precision can
        limit small movements at large coordinate magnitudes; coordinates are not implicitly
        recentered. Convergence uses residual overlap rather than stationary centers alone.
      </p>
      {!unavailableReason && (
        <p className="hint">
          Hover over a cartogram circle to inspect its computed center, radius and source identity.
          Click a circle to pin its inspection; clear it to resume hover updates.
        </p>
      )}
      <CartogramInspection analysis={analysis} />
    </section>
  );
}

/** Picking reads one computed row and keeps stale source or layout identities out of the panel. */
function CartogramInspection({analysis}: {analysis: AnalysisRuntime}) {
  const inspection = analysis.cartogramInspection;
  if (!inspection?.isCurrent()) return null;
  const {row} = inspection;
  return (
    <section aria-label="Cartogram circle inspection">
      <h4>{analysis.cartogramInspectionPinned ? 'Pinned circle' : 'Hovered circle'}</h4>
      <dl>
        <dt>Source dataset</dt>
        <dd>{inspection.datasetId}</dd>
        <dt>Source row</dt>
        <dd>{row.sourceRowIndex.toLocaleString()}</dd>
        <dt>Source ID</dt>
        <dd>{row.sourceId.toLocaleString()}</dd>
        <dt>Computed center X</dt>
        <dd>{formatMetric(row.centerX ?? NaN)} projected meters</dd>
        <dt>Computed center Y</dt>
        <dd>{formatMetric(row.centerY ?? NaN)} projected meters</dd>
        <dt>Computed circle radius</dt>
        <dd>{formatMetric(row.radius ?? NaN)} projected meters</dd>
      </dl>
      <button type="button" onClick={() => analysis.clearCartogramInspection()}>
        Clear inspection
      </button>
    </section>
  );
}

function formatMetric(value: number): string {
  return Number.isFinite(value)
    ? value.toLocaleString(undefined, {maximumSignificantDigits: 6})
    : 'undefined';
}
