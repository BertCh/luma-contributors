// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS} from '@luma-contributors/kepler';

/** Separate drafts preserve Moran's uint32 seed when switching between inference policies. */
export type PolygonQuantileDraft = {
  quantileCount: string;
  selectedQuantile: string;
  seed: string;
  permutationCount: string;
  significanceLevel: string;
};

const FIELDS = [
  {key: 'quantileCount', label: 'Polygon Quantile LISA quantile count', minimum: 2, step: '1'},
  {
    key: 'selectedQuantile',
    label: 'Polygon Quantile LISA selected quantile',
    minimum: 1,
    step: '1'
  },
  {
    key: 'seed',
    label: 'Polygon Quantile LISA seed',
    minimum: 0,
    maximum: Number.MAX_SAFE_INTEGER,
    step: '1'
  },
  {
    key: 'permutationCount',
    label: 'Polygon Quantile LISA permutations',
    minimum: 1,
    maximum: MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS,
    step: '1'
  },
  {
    key: 'significanceLevel',
    label: 'Polygon Quantile LISA significance level',
    minimum: 0,
    maximum: 1,
    step: 'any'
  }
] as const;

/** Parent-owned drafts and submission keep geometry selection and recipe restoration atomic. */
export function PolygonQuantileControls({
  value,
  onChange,
  rowCount
}: {
  value: PolygonQuantileDraft;
  onChange(value: PolygonQuantileDraft): void;
  rowCount: number;
}) {
  const quantileCount = Number(value.quantileCount);
  return (
    <fieldset className="analysis-polygon-quantile-settings" disabled={rowCount < 3}>
      <legend>Polygon Quantile LISA</legend>
      {FIELDS.map(field => (
        <label key={field.key} htmlFor={`analysis-polygon-quantile-${field.key}`}>
          {field.label}
          <input
            id={`analysis-polygon-quantile-${field.key}`}
            type="number"
            required
            min={field.minimum}
            max={
              'maximum' in field
                ? field.maximum
                : field.key === 'quantileCount'
                  ? rowCount - 1
                  : Number.isInteger(quantileCount) && quantileCount >= 2
                    ? quantileCount
                    : undefined
            }
            step={field.step}
            value={value[field.key]}
            onChange={event => onChange({...value, [field.key]: event.target.value})}
          />
        </label>
      ))}
      {rowCount < 3 && <p className="hint">Quantile LISA requires at least three source rows.</p>}
      <p className="hint">
        Type-5 quantiles include the lower boundary and exclude the upper boundary. Tied values stay
        together and can leave quantiles empty. Missing numeric values and rows without eligible
        polygon parts are omitted before quantiling. Isolated eligible polygons can remain in the
        quantile distribution, but are excluded from inference.
      </p>
      <p className="hint">
        This compatibility statistic uses legacy directed pseudo p-values: (min(upper-tail count,
        permutations − upper-tail count) + 1) / (permutations + 1). It is anti-conservative, has no
        two-sided error-control guarantee and applies no multiple-testing correction. Tied or
        constant memberships can appear significant. Treat results as exploratory.
      </p>
      <p className="hint">
        Only selected-quantile members with positive local joins are tested. Selection recomputes
        quantile boundaries and filters the fixed source-wide contiguity graph without changing its
        topology. More than 64 active neighbors per row makes inference unavailable, even when the
        graph fits its storage capacity.
      </p>
      <p className="hint">
        Gray includes nonsignificant and untested rows, red marks significance under the legacy
        rule, and orange marks islands. The table retains membership, join count, pseudo p-value and
        legacy tail count. A fixed seed reproduces this implementation for the same source and
        population; GeoDa random-stream parity is not claimed. Significance must remain strictly
        between 0 and 1 in Float32.
      </p>
    </fieldset>
  );
}
