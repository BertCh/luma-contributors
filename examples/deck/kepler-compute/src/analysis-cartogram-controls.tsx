// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MAXIMUM_KEPLER_DORLING_SOURCE_ROWS,
  MAXIMUM_KEPLER_DORLING_ITERATIONS
} from '@luma-contributors/kepler';

/** Parent-owned strings preserve incomplete edits until the full layout is submitted. */
export type CartogramDraft = {
  xColumn: string;
  yColumn: string;
  valueColumn: string;
  crs: string;
  minX: string;
  minY: string;
  maxX: string;
  maxY: string;
  areaScale: string;
  maximumIterations: string;
  overlapTolerance: string;
  attraction: string;
};

const COLUMN_FIELDS = [
  {key: 'xColumn', label: 'Cartogram projected X field'},
  {key: 'yColumn', label: 'Cartogram projected Y field'},
  {key: 'valueColumn', label: 'Cartogram nonnegative area-value field'}
] as const;

const NUMERIC_FIELDS = [
  {key: 'minX', label: 'Cartogram minimum X in projected meters', step: 'any'},
  {key: 'minY', label: 'Cartogram minimum Y in projected meters', step: 'any'},
  {key: 'maxX', label: 'Cartogram maximum X in projected meters', step: 'any'},
  {key: 'maxY', label: 'Cartogram maximum Y in projected meters', step: 'any'},
  {
    key: 'areaScale',
    label: 'Cartogram area scale in projected square meters per value',
    minimum: 0,
    step: 'any'
  },
  {
    key: 'maximumIterations',
    label: 'Cartogram maximum iterations',
    minimum: 0,
    maximum: MAXIMUM_KEPLER_DORLING_ITERATIONS,
    step: '1'
  },
  {
    key: 'overlapTolerance',
    label: 'Cartogram overlap tolerance in projected meters',
    minimum: 0,
    step: 'any'
  },
  {key: 'attraction', label: 'Cartogram attraction', minimum: 0, maximum: 1, step: 'any'}
] as const;

/** A bounded Dorling layout is a separate analysis result, not a replacement map geometry. */
export function CartogramControls({
  draft,
  onChange,
  columns,
  rowCount
}: {
  draft: CartogramDraft;
  onChange(draft: CartogramDraft): void;
  columns: readonly string[];
  rowCount: number;
}) {
  const unsupportedRowCount = rowCount < 1 || rowCount > MAXIMUM_KEPLER_DORLING_SOURCE_ROWS;
  return (
    <fieldset className="analysis-cartogram-settings" disabled={unsupportedRowCount}>
      <legend>Dorling circle cartogram</legend>
      {COLUMN_FIELDS.map(field => (
        <label key={field.key} htmlFor={`analysis-cartogram-${field.key}`}>
          {field.label}
          <select
            id={`analysis-cartogram-${field.key}`}
            aria-label={field.label}
            required
            value={draft[field.key]}
            onChange={event => onChange({...draft, [field.key]: event.target.value})}
          >
            <option value="">Choose a prepared Float32 field</option>
            {columns.map(column => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>
      ))}
      <label htmlFor="analysis-cartogram-crs">
        Cartogram declared projected CRS
        <input
          id="analysis-cartogram-crs"
          type="text"
          required
          placeholder="For example, EPSG:3857"
          value={draft.crs}
          onChange={event => onChange({...draft, crs: event.target.value})}
        />
      </label>
      {NUMERIC_FIELDS.map(field => (
        <label key={field.key} htmlFor={`analysis-cartogram-${field.key}`}>
          {field.label}
          <input
            id={`analysis-cartogram-${field.key}`}
            type="number"
            required
            min={'minimum' in field ? field.minimum : undefined}
            max={'maximum' in field ? field.maximum : undefined}
            step={field.step}
            value={draft[field.key]}
            onChange={event => onChange({...draft, [field.key]: event.target.value})}
          />
        </label>
      ))}
      {unsupportedRowCount && (
        <p className="hint">
          Cartograms support 1 to {MAXIMUM_KEPLER_DORLING_SOURCE_ROWS.toLocaleString()} source rows.
        </p>
      )}
      <p className="hint">
        X and Y must already be projected coordinates in meter units. Declare the source CRS and
        finite bounds covering those coordinates. This form does not project longitude and latitude,
        convert units, or replace source geometry. The CRS declaration records the coordinate
        meaning; it does not verify the source projection.
      </p>
      <p className="hint">
        Each eligible row becomes a circle whose area is its nonnegative value multiplied by the
        nonnegative area scale. Area uses projected square meters and overlap tolerance uses
        projected meters; these are not ground area or ground distance. A zero-area value remains a
        stationary circle with zero radius. Missing or nonfinite coordinates or values, and negative
        area values, are excluded while retaining source row identity.
      </p>
      <p className="hint">
        Dorling placement is an iterative heuristic. Attraction pulls circles toward their original
        positions while the layout reduces overlap. Choose 0 to 256 iterations; zero returns the
        initial layout without relaxation and may exhaust the budget with overlap remaining. A
        finite layout is still available when the iteration budget is exhausted; inspect its overlap
        and convergence diagnostics before interpreting the result. Convergence is not guaranteed.
      </p>
      <p className="hint">
        Arbitrary declared projected CRSs produce table results. An EPSG:3857 map overlay requires a
        validated map projection contract; the layout must not be interpreted as geographic map
        coordinates. Float32 positions and calculations limit precision, particularly for large
        coordinate magnitudes or small circles. Coordinates are not implicitly recentered; choose an
        overlap tolerance in projected meters that is meaningful at the retained Float32 precision.
      </p>
    </fieldset>
  );
}
