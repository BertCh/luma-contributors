// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PointCoordinateDraft} from './analysis-point-coordinate-controls';
import {MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS} from '@luma-contributors/kepler';
import {
  normalizeQuantileLisaSettings,
  type QuantileLisaSettings
} from './analysis-quantile-lisa-inference';
import {ResidentAnalysisSheetForm} from './analysis-sheet';
import {
  makePointSheetDefaults,
  usePointAnalysisSheet,
  createPointSheetFields,
  getPointSheetTopology,
  getPointSheetChecks,
  type PointSheetValues
} from './analysis-point-weights-panel';

export function getQuantileLisaSheetSettings(
  values: PointSheetValues,
  rowCount: number
): QuantileLisaSettings {
  return normalizeQuantileLisaSettings(
    {
      ...getPointSheetTopology(values, rowCount),
      quantileCount: Number(values['quantileCount']),
      selectedQuantile: Number(values['selectedQuantile']),
      seed: Number(values['seed']),
      permutationCount: Number(values['permutationCount']),
      significanceLevel: Number(values['significanceLevel']),
      alternative: values['alternative'] as QuantileLisaSettings['alternative']
    },
    rowCount
  );
}

export function QuantileLisaPanel({
  columns,
  rowCount,
  settings,
  coordinateInitialization,
  coordinateResetKey,
  coordinateSource,
  onApply
}: {
  coordinateInitialization?: PointCoordinateDraft;
  coordinateResetKey?: unknown;
  coordinateSource?: string;
  columns: readonly string[];
  rowCount: number;
  settings?: QuantileLisaSettings;
  onApply(settings: QuantileLisaSettings): void;
}) {
  const sheet = usePointAnalysisSheet(
    makePointSheetDefaults(settings, coordinateInitialization, rowCount, {
      quantileCount: settings?.quantileCount ?? 2,
      selectedQuantile: settings?.selectedQuantile ?? 2,
      seed: settings?.seed ?? 314159,
      permutationCount: settings?.permutationCount ?? 199,
      significanceLevel: settings?.significanceLevel ?? 0.05,
      alternative: settings?.alternative ?? 'geoda-legacy'
    }),
    settings,
    coordinateResetKey,
    coordinateSource
  );
  const fields = createPointSheetFields(
    sheet.values,
    columns,
    'analysis-quantile-lisa',
    'Quantile LISA',
    sheet.source,
    [
      {
        name: 'alternative',
        label: 'Quantile LISA p-value alternative',
        controlId: 'analysis-quantile-lisa-alternative',
        kind: 'enum',
        options: ['greater', 'two-sided', 'geoda-legacy']
      },
      {
        name: 'quantileCount',
        label: 'Quantile LISA quantile count',
        controlId: 'analysis-quantile-lisa-quantileCount',
        kind: 'integer',
        minimum: 2,
        maximum: rowCount - 1
      },
      {
        name: 'selectedQuantile',
        label: 'Quantile LISA selected quantile',
        controlId: 'analysis-quantile-lisa-selectedQuantile',
        kind: 'integer',
        minimum: 1,
        maximum: Number(sheet.values['quantileCount'])
      },
      {
        name: 'seed',
        label: 'Quantile LISA seed',
        controlId: 'analysis-quantile-lisa-seed',
        kind: 'integer',
        minimum: 0,
        maximum: Number.MAX_SAFE_INTEGER,
        level: 'advanced'
      },
      {
        name: 'permutationCount',
        label: 'Quantile LISA permutations',
        controlId: 'analysis-quantile-lisa-permutationCount',
        kind: 'integer',
        minimum: 1,
        maximum: MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS,
        level: 'advanced'
      },
      {
        name: 'significanceLevel',
        label: 'Quantile LISA significance level',
        controlId: 'analysis-quantile-lisa-significanceLevel',
        kind: 'number',
        minimum: 0,
        maximum: 1,
        level: 'advanced'
      }
    ],
    {
      longitudeId: 'analysis-quantile-lisa-longitudeColumn',
      latitudeId: 'analysis-quantile-lisa-latitudeColumn',
      nearestId: 'analysis-quantile-lisa-k',
      permutations: true
    }
  );
  return (
    <details className="analysis-quantile-lisa-settings">
      <summary>
        Configure Quantile LISA on{' '}
        {sheet.values['coordinateSpace'] === 'planar' ? 'projected' : 'geographic'} points
      </summary>
      <ResidentAnalysisSheetForm
        key={JSON.stringify([coordinateResetKey, settings])}
        fields={fields}
        checks={getPointSheetChecks(fields, () =>
          getQuantileLisaSheetSettings(sheet.values, rowCount)
        )}
        method="Resident Quantile LISA counts binary-weight neighbors in a selected quantile. Choose the p-value alternative explicitly; no multiple-testing correction is applied."
        submitLabel="Apply neighbors and run Quantile LISA"
        onChange={(name, value) => sheet.change(name, value)}
        onApply={current =>
          onApply(getQuantileLisaSheetSettings({...sheet.values, ...current}, rowCount))
        }
      />
      <p className="hint">
        Type-5 quantiles include the lower boundary and exclude the upper boundary. Ties stay
        together and can leave quantiles empty. Missing numeric values and invalid coordinates are
        omitted before quantiling. Selection recomputes boundaries and removes neighbors from the
        fixed source graph without replacements.
      </p>
      <p className="hint">
        Greater uses the inclusive upper tail. Two-sided uses inclusive tails and gives p = 1 for
        all ties. The geoda-legacy compatibility choice uses (min(upper-tail count, permutations −
        upper-tail count) + 1) / (permutations + 1). It is anti-conservative, has no two-sided
        error-control guarantee and applies no multiple-testing correction. Tied or constant
        memberships can appear significant. Treat results as exploratory.
      </p>
      <p className="hint">
        Only selected-quantile members with positive local joins are tested. Isolated points are
        excluded from inference. Gray includes nonsignificant and untested rows; red denotes
        significance under the chosen alternative; orange denotes isolated points. The table retains
        membership, join count, pseudo p-value and tail count. A fixed seed is reproducible for the
        same source and selection. More rows, neighbors and permutations increase computation time.
      </p>
    </details>
  );
}
