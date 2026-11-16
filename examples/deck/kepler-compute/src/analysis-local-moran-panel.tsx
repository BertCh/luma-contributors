// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PointCoordinateDraft} from './analysis-point-coordinate-controls';
import {
  normalizeLocalMoranSettings,
  type LocalMoranSettings
} from './analysis-local-moran-settings';
import {ResidentAnalysisSheetForm} from './analysis-sheet';
import {
  makePointSheetDefaults,
  usePointAnalysisSheet,
  createPointSheetFields,
  getPointSheetTopology,
  getPointSheetChecks,
  makePointSheetColumn,
  POINT_SHEET_CORRECTIONS,
  type PointSheetValues,
  type PointSheetSpecification
} from './analysis-point-weights-panel';

export function getLocalMoranSheetSettings(
  values: PointSheetValues,
  rowCount: number
): LocalMoranSettings {
  const multivariate = values['mode'] === 'multivariate';
  const bivariate = values['mode'] === 'bivariate';
  const inference =
    bivariate || multivariate || values['inference'] === 'permutation-conditional-two-sided'
      ? 'permutation-conditional-two-sided'
      : 'analytic-conditional-normal-two-sided';
  return normalizeLocalMoranSettings(
    {
      ...getPointSheetTopology(values, rowCount),
      minimumNeighbors:
        inference === 'permutation-conditional-two-sided' ? 1 : Number(values['minimumNeighbors']),
      significanceLevel: Number(values['significanceLevel']),
      islands: values['islands'] as LocalMoranSettings['islands'],
      multipleTesting: values['multipleTesting'] as LocalMoranSettings['multipleTesting'],
      inference,
      ...(bivariate ? {secondColumn: String(values['secondColumn'] ?? '')} : {}),
      ...(multivariate
        ? {
            variableColumns: Array.from({length: Number(values['variableCount'])}, (_, index) =>
              String(values[`variableColumn${index}`] ?? '')
            )
          }
        : {}),
      ...(inference === 'permutation-conditional-two-sided'
        ? {permutationCount: Number(values['permutationCount']), seed: Number(values['seed'])}
        : {})
    },
    rowCount
  );
}

/** Every supported resident method retains its own weights and inference settings. */
export function LocalMoranPanel({
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
  settings?: LocalMoranSettings;
  onApply(settings: LocalMoranSettings): void;
}) {
  const sheet = usePointAnalysisSheet(
    makePointSheetDefaults(settings, coordinateInitialization, rowCount, {
      mode: settings?.variableColumns?.length
        ? 'multivariate'
        : settings?.secondColumn
          ? 'bivariate'
          : 'univariate',
      minimumNeighbors: settings?.minimumNeighbors ?? 4,
      significanceLevel: settings?.significanceLevel ?? 0.05,
      islands: settings?.islands ?? 'exclude',
      multipleTesting: settings?.multipleTesting ?? 'benjamini-hochberg',
      inference: settings?.inference ?? 'analytic-conditional-normal-two-sided',
      permutationCount: settings?.permutationCount ?? 999,
      seed: settings?.seed ?? 0,
      secondColumn: settings?.secondColumn ?? columns[0] ?? '',
      variableCount: settings?.variableColumns?.length ?? 2,
      ...Object.fromEntries(
        Array.from({length: 8}, (_, index) => [
          `variableColumn${index}`,
          settings?.variableColumns?.[index] ?? columns[index] ?? ''
        ])
      )
    }),
    settings,
    coordinateResetKey,
    coordinateSource
  );
  const multivariate = sheet.values['mode'] === 'multivariate';
  const bivariate = sheet.values['mode'] === 'bivariate';
  const permutationInference =
    bivariate || multivariate || sheet.values['inference'] === 'permutation-conditional-two-sided';
  const parameters: PointSheetSpecification[] = [
    {
      name: 'mode',
      label: 'Local Moran variables',
      controlId: 'analysis-moran-mode',
      kind: 'enum',
      options: ['univariate', 'bivariate', 'multivariate'],
      description:
        'Univariate uses the selected numeric field; bivariate compares it with the lag of a second field; multivariate computes local Geary over 2 to 8 ordered fields.'
    },
    ...(bivariate
      ? [
          makePointSheetColumn(
            'secondColumn',
            'Local Moran second variable',
            'analysis-moran-second-column'
          )
        ]
      : []),
    ...(multivariate
      ? [
          {
            name: 'variableCount',
            label: 'Local Geary variable count',
            controlId: 'analysis-moran-variable-count',
            kind: 'integer' as const,
            minimum: 2,
            maximum: 8
          },
          ...Array.from({length: Number(sheet.values['variableCount'])}, (_, index) =>
            makePointSheetColumn(
              `variableColumn${index}`,
              `Local Geary variable ${index + 1}`,
              `analysis-moran-variable-${index}`
            )
          )
        ]
      : []),
    ...(!bivariate && !multivariate
      ? [
          {
            name: 'inference',
            label: 'Local Moran inference',
            controlId: 'analysis-moran-inference',
            kind: 'enum' as const,
            options: ['analytic-conditional-normal-two-sided', 'permutation-conditional-two-sided']
          }
        ]
      : []),
    ...(permutationInference
      ? [
          {
            name: 'permutationCount',
            label: 'Local Moran permutations',
            controlId: 'analysis-moran-permutations',
            kind: 'integer' as const,
            minimum: 1,
            maximum: 9999,
            level: 'advanced' as const
          },
          {
            name: 'seed',
            label: 'Local Moran permutation seed',
            controlId: 'analysis-moran-seed',
            kind: 'integer' as const,
            minimum: 0,
            maximum: 4294967295,
            level: 'advanced' as const
          }
        ]
      : [
          {
            name: 'minimumNeighbors',
            label: 'Local Moran minimum neighbors',
            controlId: 'analysis-moran-minimum',
            kind: 'integer' as const,
            minimum: 1,
            maximum: 2 ** 24 - 1
          }
        ]),
    {
      name: 'significanceLevel',
      label: 'Local Moran significance level',
      controlId: 'analysis-moran-significance',
      kind: 'number',
      minimum: 0,
      maximum: 1,
      level: 'advanced'
    },
    {
      name: 'islands',
      label: 'Local Moran isolated points',
      controlId: 'analysis-moran-islands',
      kind: 'enum',
      options: ['exclude', 'keep'],
      level: 'advanced'
    },
    {
      name: 'multipleTesting',
      label: 'Local Moran correction',
      controlId: 'analysis-moran-correction',
      kind: 'enum',
      options: multivariate ? ['none', 'bonferroni'] : POINT_SHEET_CORRECTIONS,
      level: 'advanced'
    }
  ];
  const fields = createPointSheetFields(
    sheet.values,
    columns,
    'analysis-moran',
    'Local Moran',
    sheet.source,
    parameters,
    {permutations: permutationInference}
  );
  return (
    <details className="analysis-local-moran-settings">
      <summary>
        Configure Local Moran on{' '}
        {sheet.values['coordinateSpace'] === 'planar' ? 'projected' : 'geographic'} points
      </summary>
      <ResidentAnalysisSheetForm
        key={JSON.stringify([coordinateResetKey, settings])}
        fields={fields}
        checks={getPointSheetChecks(fields, () =>
          getLocalMoranSheetSettings(sheet.values, rowCount)
        )}
        method="Resident Local Moran uses binary weights and defaults to analytic conditional-normal inference. Catalog Local Moran defaults to row-standardized weights, 999 permutations and Benjamini–Hochberg correction. Bivariate Local Moran and multivariate local Geary require conditional permutations with minimum neighbors fixed at 1."
        submitLabel="Apply neighbors and run Local Moran"
        onChange={(name, value) => {
          const changes: Record<string, string | number> = {};
          if (
            (name === 'mode' && value !== 'univariate') ||
            (name === 'inference' && value === 'permutation-conditional-two-sided')
          ) {
            changes['inference'] = 'permutation-conditional-two-sided';
            changes['minimumNeighbors'] = 1;
          }
          if (
            name === 'mode' &&
            value === 'multivariate' &&
            sheet.values['multipleTesting'] !== 'none' &&
            sheet.values['multipleTesting'] !== 'bonferroni'
          )
            changes['multipleTesting'] = 'none';
          sheet.change(name, value, changes);
        }}
        onApply={current =>
          onApply(getLocalMoranSheetSettings({...sheet.values, ...current}, rowCount))
        }
      />
      {bivariate && (
        <p className="hint">
          Bivariate Local Moran compares the selected field at each point with the weighted neighbor
          average of the second field. Only conditional permutation inference is available; outputs
          retain the statistic, p-values, quadrants and clusters.
        </p>
      )}
      {multivariate && (
        <p className="hint">
          Variables are standardized jointly and retain their selected order. Conditional
          permutations draw whole neighbor vectors. Outputs retain the mean local Geary statistic,
          p-value and similarity/dissimilarity class. Use at most 9999 permutations with none or
          Bonferroni correction; rows with more than 64 active neighbors remain untested.
        </p>
      )}
      {permutationInference ? (
        <>
          <p className="hint">
            Two-sided conditional randomization keeps the focus value fixed and samples distinct
            other observations for its active neighbor slots. With P permutations, the raw p-value
            resolution has a floor of 2 / (P + 1); all-tied simulations give p = 1. Multiple-testing
            corrections can require a larger permutation count to resolve significance.
          </p>
          <p className="hint">
            The table z-score remains a separate analytic diagnostic. Permutation p-values and their
            corrected values determine the displayed quadrants. The seed reproduces this
            implementation's draws; GeoDa random-stream parity is not claimed.
          </p>
          <p className="hint">
            Permutation inference supports fewer than 16.7 million source rows (tiled GPU
            dispatches, results independent of tiling) and 9,999 permutations. Minimum neighbors is
            fixed at 1; rows with no included neighbors are not tested.
          </p>
        </>
      ) : (
        <p className="hint">
          Inference uses a two-sided normal approximation to the conditional randomization moments,
          without permutations. Small neighborhoods and skewed values can make this approximation
          poor. The table z-score is the analytic conditional-normal diagnostic.
        </p>
      )}
      <p className="hint">
        Quadrants identify high–high, low–high, low–low and high–low association only where the
        corrected p-value meets the significance level. Isolated points have no inferred quadrant;
        keeping them retains their values in global moments with local I and spatial lag zero.
      </p>
      <p className="hint">
        Weights stay binary and are not normalized to sum to one. Selection filters the fixed
        source-wide point graph without finding replacement neighbors. Constant values, too few
        neighbors, or undefined variance have no analytic score.
      </p>
    </details>
  );
}
