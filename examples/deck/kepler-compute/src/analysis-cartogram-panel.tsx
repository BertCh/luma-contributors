// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {
  MAXIMUM_KEPLER_DORLING_SOURCE_ROWS,
  parseKeplerDorlingSettings,
  type KeplerDorlingSettings
} from '@luma-contributors/kepler';
import type {AnalysisRuntime} from './analysis-runtime';
import {CartogramControls, type CartogramDraft} from './analysis-cartogram-controls';
import {CartogramDeclarationForm} from './analysis2-cartogram-declaration-form';
import {ResidentConsumerNotice} from './resident-consumer-notice';

function makeDraft(
  settings: KeplerDorlingSettings | undefined,
  valueColumn: string
): CartogramDraft {
  return {
    xColumn: settings?.xColumn ?? '',
    yColumn: settings?.yColumn ?? '',
    valueColumn: settings?.valueColumn ?? valueColumn,
    crs: settings?.crs ?? '',
    minX: settings ? String(settings.bounds[0]) : '',
    minY: settings ? String(settings.bounds[1]) : '',
    maxX: settings ? String(settings.bounds[2]) : '',
    maxY: settings ? String(settings.bounds[3]) : '',
    areaScale: String(settings?.areaScale ?? 1),
    maximumIterations: String(settings?.maximumIterations ?? 100),
    overlapTolerance: String(settings?.overlapTolerance ?? 0.001),
    attraction: String(settings?.attraction ?? 0.01)
  };
}

/** A saved cartogram owns its value and coordinate roles independently of the general numeric field. */
export function CartogramPanel({analysis}: {analysis: AnalysisRuntime}) {
  const settings = analysis.settings.cartogram;
  const valueColumn = settings?.valueColumn ?? analysis.settings.column;
  const [draft, setDraft] = useState(() => makeDraft(settings, valueColumn));
  const [error, setError] = useState('');
  useEffect(() => {
    setDraft(makeDraft(settings, valueColumn));
    setError('');
  }, [
    analysis.datasetId,
    settings?.xColumn,
    settings?.yColumn,
    valueColumn,
    settings?.crs,
    settings?.bounds[0],
    settings?.bounds[1],
    settings?.bounds[2],
    settings?.bounds[3],
    settings?.areaScale,
    settings?.maximumIterations,
    settings?.overlapTolerance,
    settings?.attraction
  ]);
  return (
    <details className="analysis-cartogram-panel">
      <summary>Configure Dorling circle cartogram</summary>
      <p className="hint">
        Choose the value that determines circle area independently of other analyses. The current
        analysis population controls which source rows participate in the layout.
      </p>
      <form
        onSubmit={event => {
          event.preventDefault();
          try {
            if (
              [draft.xColumn, draft.yColumn, draft.valueColumn].some(
                column => !column || !analysis.availableColumns.includes(column)
              )
            )
              throw new Error('Choose prepared Float32 fields for X, Y and circle area values');
            if (
              [
                draft.minX,
                draft.minY,
                draft.maxX,
                draft.maxY,
                draft.areaScale,
                draft.maximumIterations,
                draft.overlapTolerance,
                draft.attraction
              ].some(value => !value.trim())
            )
              throw new Error('Declare all cartogram bounds and layout parameters');
            const next = parseKeplerDorlingSettings({
              coordinateSpace: 'planar',
              metric: 'native',
              units: 'meters',
              xColumn: draft.xColumn,
              yColumn: draft.yColumn,
              valueColumn: draft.valueColumn,
              crs: draft.crs,
              bounds: [
                Number(draft.minX),
                Number(draft.minY),
                Number(draft.maxX),
                Number(draft.maxY)
              ],
              areaScale: Number(draft.areaScale),
              maximumIterations: Number(draft.maximumIterations),
              overlapTolerance: Number(draft.overlapTolerance),
              attraction: Number(draft.attraction)
            });
            analysis.configureCartogram(next);
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        <CartogramControls
          draft={draft}
          onChange={setDraft}
          columns={analysis.availableColumns}
          rowCount={analysis.rowCount}
        />
        <button
          type="submit"
          disabled={analysis.rowCount < 1 || analysis.rowCount > MAXIMUM_KEPLER_DORLING_SOURCE_ROWS}
        >
          Apply fields and run cartogram
        </button>
      </form>
      {settings && settings.crs !== 'EPSG:3857' && <CartogramDeclarationForm analysis={analysis} />}
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
