// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateFiniteNumber, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  getMultipleTestingAdjustedPValues,
  summarizeGPUAnalysisResult
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogOperationBinding, CatalogRunResult} from '../analysis/catalog-binding-types';
import type {KeplerOperationDescriptor} from '../descriptors/operation-descriptor-types';
import {getKeplerBindingResultPresentations, getKeplerResultToneColor} from './result-presentation';
import {describeKeplerCatalogResult} from './result-report';
import type {KeplerResultReport, KeplerResultReportContext} from './result-report';
import type {KeplerBindingResultPresentation} from './result-presentation';

/** Statistical reports include table-only global tests as well as mapped significance classes. */
export function hasKeplerResultInference(
  report: KeplerResultReport,
  presentations: readonly KeplerBindingResultPresentation[] = []
): boolean {
  const statisticalTemplates = [
    'global-statistic',
    'local-cluster-count',
    'significant-hot-cold-count'
  ];
  return (
    report.outputs.some(
      output =>
        output.inference ??
        (statisticalTemplates.includes(output.template) ||
          statisticalTemplates.includes(output.usedTemplate))
    ) ||
    presentations.some(presentation =>
      ['hot-spot-confidence-bins', 'local-moran-clusters', 'significance-flag'].includes(
        presentation.presentation?.classScheme ?? ''
      )
    )
  );
}

/** Corrections supported by both the retained-statistics adapter and the original bindings. */
export type KeplerResultInferenceSettings = {
  significance: number;
  multipleTesting: 'none' | 'benjamini-hochberg' | 'bonferroni';
};

export type KeplerResultInferenceScope = 'full' | 'sample' | 'viewport';

export type KeplerResultInferenceUpdate = Extract<
  ReturnType<typeof applyKeplerResultInference>,
  {status: 'display'}
>;

/** Values shown initially; these are display settings, never a replacement for run provenance. */
export function getKeplerResultInferenceSettings(
  result: CatalogRunResult
): KeplerResultInferenceSettings {
  const correction = result.parameters['multipleTesting'];
  return {
    significance:
      result.operationId === 'recipe-hot-spot-lattice'
        ? 0.1
        : Number(result.parameters['significance'] ?? 0.05),
    multipleTesting:
      correction === 'benjamini-hochberg' || correction === 'bonferroni' ? correction : 'none'
  };
}

/**
 * Reclassify the entire retained testing family without executing a statistic or reading the GPU.
 * Only verified binding layouts are accepted. An adjusted column can support a new cut-off with
 * the same correction; changing the correction requires the raw column. No inversion is attempted.
 * The original run and all its arrays remain untouched. Hosts must atomically accept tables, plans
 * and report together; `displayResult` is a view, and must not be serialized as an executed run.
 */
export function applyKeplerResultInference(
  result: CatalogRunResult,
  binding: CatalogOperationBinding,
  descriptor: KeplerOperationDescriptor,
  settings: KeplerResultInferenceSettings,
  context: KeplerResultReportContext = {},
  scope: KeplerResultInferenceScope = 'full'
) {
  const identifier = 'applyKeplerResultInference';
  // Match the Local Moran binding's probability limit; hot spots have narrower fixed bands below.
  const MAXIMUM_SIGNIFICANCE = 1;
  validateFiniteNumber(identifier, 'significance', settings.significance);
  validateRequirement(
    identifier,
    settings.significance > 0 && settings.significance <= MAXIMUM_SIGNIFICANCE,
    'significance must be greater than 0 and at most 1'
  );
  validateRequirement(
    identifier,
    ['none', 'benjamini-hochberg', 'bonferroni'].includes(settings.multipleTesting),
    'multipleTesting must be none, benjamini-hochberg or bonferroni'
  );
  const rerun = (reason: string) => ({status: 'rerun-required' as const, settings, reason});
  if (scope !== 'full')
    return rerun('Recompute on the full population before displaying significance classes.');
  if (binding.id !== result.operationId || descriptor.bindingId !== result.operationId) {
    return rerun('The binding and descriptor must describe the retained run.');
  }
  const local = result.operationId === 'local-moran-statistics';
  const hot = result.operationId === 'recipe-hot-spot-lattice';
  if (!local && !hot)
    return rerun(
      'This operation has no verified retained-statistics adapter; rerun with the requested inference settings.'
    );
  if (hot && settings.multipleTesting === 'bonferroni')
    return rerun('The lattice hot-spot binding supports none or Benjamini-Hochberg correction.');
  if (hot && settings.significance > 0.1)
    return rerun(
      'The fixed hot-spot classes support cut-offs at most 0.1. Choose a smaller cut-off.'
    );
  const tableId = local ? 'local-moran' : 'recipe-hot-spots';
  const table = result.tables.find(candidate => candidate.id === tableId);
  if (!table) return rerun(`The retained ${tableId} table is missing.`);
  const columns = table.columns;
  const hasColumn = (name: string) => columns[name]?.length === table.rowCount;
  if (
    !hasColumn('neighbor_count') ||
    (local && result.summary['featureCount'] !== table.rowCount)
  ) {
    return rerun(
      'The complete analysed population and neighbour counts must be retained; rerun to recover them.'
    );
  }
  if (hot) {
    const latticeColumns = result.summary['columns'];
    const latticeRows = result.summary['rows'];
    const testedCellCount = result.summary['testedCellCount'];
    const latticePopulation = Number(latticeColumns) * Number(latticeRows);
    const cellIndices = columns['cell_index'];
    const scores = columns['z_score'];
    if (
      typeof latticeColumns !== 'number' ||
      !Number.isSafeInteger(latticeColumns) ||
      latticeColumns <= 0 ||
      typeof latticeRows !== 'number' ||
      !Number.isSafeInteger(latticeRows) ||
      latticeRows <= 0 ||
      !Number.isSafeInteger(latticePopulation) ||
      table.rowCount !== latticePopulation ||
      !hasColumn('cell_index') ||
      !hasColumn('z_score') ||
      new Set(cellIndices).size !== latticePopulation ||
      cellIndices.some(
        value =>
          typeof value !== 'number' ||
          !Number.isSafeInteger(value) ||
          value < 0 ||
          value >= latticePopulation
      ) ||
      typeof testedCellCount !== 'number' ||
      !Number.isSafeInteger(testedCellCount) ||
      testedCellCount !==
        scores.filter(value => typeof value === 'number' && Number.isFinite(value)).length
    ) {
      return rerun(
        'The complete original lattice population, cell identities and tested count must be retained; rerun to recover them.'
      );
    }
  }
  const rawAvailable = hasColumn('p_value');
  const originalCorrection = getKeplerResultInferenceSettings(result).multipleTesting;
  if (
    !rawAvailable &&
    (!hasColumn('adjusted_p_value') || settings.multipleTesting !== originalCorrection)
  ) {
    return rerun(
      'Raw p-values were not retained. Rerun to change correction; adjusted p-values cannot recover them.'
    );
  }
  const finite = (value: unknown): value is number =>
    typeof value === 'number' && Number.isFinite(value);
  const probability = (value: unknown): value is number =>
    finite(value) && value >= 0 && value <= 1;
  const probabilities = columns[rawAvailable ? 'p_value' : 'adjusted_p_value'];
  // Exclusion belongs to the statistic's population, never the current viewport or selection.
  const excludedRows = probabilities.flatMap((value, row) =>
    !probability(value) || columns['neighbor_count']?.[row] === 0 ? [row] : []
  );
  if (excludedRows.length === table.rowCount && table.rowCount > 0) {
    return rerun(
      'No retained p-values are testable. Rerun with inference enabled (for example, 999 permutations).'
    );
  }
  const excluded = new Set(excludedRows);
  const directions = local ? ['value', 'spatial_lag'] : ['z_score'];
  if (
    !directions.every(hasColumn) ||
    probabilities.some(
      (_, row) => !excluded.has(row) && directions.some(name => !finite(columns[name][row]))
    )
  ) {
    return rerun(
      'Class direction was not retained. Rerun to recover the statistic and spatial lag.'
    );
  }
  // Local Moran publishes every row in its analysis set; its value mean recovers the centred sign,
  // including rows whose permutation test was excluded. Incomplete values cannot recover it.
  if (local && columns['value'].some(value => !finite(value)))
    return rerun(
      'The complete analysed value population is required to recover local Moran quadrants.'
    );
  const mean = local
    ? (columns['value'] as number[]).reduce((total, value) => total + value, 0) / table.rowCount
    : 0;
  const adjusted = rawAvailable
    ? Array.from(
        getMultipleTestingAdjustedPValues(
          probabilities.map((value, row) => (excluded.has(row) ? Number.NaN : Number(value))),
          settings.multipleTesting
        )
      )
    : probabilities.map(value => Number(value));
  const classes = adjusted.map((value, row) => {
    if (excluded.has(row)) return local ? 'not-tested' : null;
    if (value > settings.significance) return local ? 'not-significant' : 0;
    if (local)
      return Number(columns['value'][row]) >= mean
        ? Number(columns['spatial_lag'][row]) >= 0
          ? 'HH'
          : 'HL'
        : Number(columns['spatial_lag'][row]) >= 0
          ? 'LH'
          : 'LL';
    const level = value <= 0.01 ? 3 : value <= 0.05 ? 2 : 1;
    return Number(columns['z_score'][row]) < 0 ? -level : level;
  });
  const displayTable = {
    ...table,
    columns: {
      ...columns,
      adjusted_p_value: adjusted.map((value, row) => (excluded.has(row) ? null : value)),
      [local ? 'cluster' : 'bin']: classes
    },
    columnTypes: {...table.columnTypes, adjusted_p_value: 'float64' as const}
  };
  const displayResult = {
    ...result,
    tables: result.tables.map(candidate => (candidate === table ? displayTable : candidate)),
    parameters: {...result.parameters, ...settings},
    summary: {
      ...result.summary,
      multipleTesting: settings.multipleTesting,
      ...(hot ? {testedCellCount: table.rowCount - excludedRows.length} : {})
    }
  };
  const presentations = getKeplerBindingResultPresentations(displayResult, descriptor).map(
    presentation => {
      const plan = presentation.presentation;
      if (presentation.tableId !== tableId || local || !excludedRows.length || !plan)
        return presentation;
      // The shared hot-spot plan identifies islands; also exclude missing/invalid inference rows.
      const legend = [
        ...plan.legend.filter(entry => entry.tone !== 'excluded'),
        {
          value: 'excluded-untested',
          label: 'Untested',
          tone: 'excluded' as const,
          count: excludedRows.length,
          color: getKeplerResultToneColor('excluded')
        }
      ];
      return {
        ...presentation,
        presentation: {...plan, excludedRows, missingCount: 0, unrecognized: 0, legend},
        legend: legend.map(({label, color, count}) => ({label, color, count}))
      };
    }
  );
  const originalReport = describeKeplerCatalogResult(result, binding, descriptor, context);
  const generatedReport = describeKeplerCatalogResult(displayResult, binding, descriptor, context);
  const hotCount = classes.filter(value => typeof value === 'number' && value > 0).length;
  const coldCount = classes.filter(value => typeof value === 'number' && value < 0).length;
  const sentence = local
    ? generatedReport.sentence
    : summarizeGPUAnalysisResult('significant-hot-cold-count', {
        featureCount: table.rowCount - excludedRows.length,
        hotCount,
        coldCount,
        confidencePercent: (1 - settings.significance) * 100,
        correction:
          settings.multipleTesting === 'benjamini-hochberg' ? 'false-discovery-rate' : 'none',
        featureNoun: {singular: 'cell', plural: 'cells'}
      });
  const report = {
    ...generatedReport,
    sentence,
    outputs: generatedReport.outputs.map(output =>
      output.tableId === tableId ? {...output, sentence} : output
    ),
    chosen: originalReport.chosen,
    notes: [
      ...originalReport.notes,
      `Display cut-off: ${settings.significance}; correction: ${settings.multipleTesting}. The statistic was not rerun.`
    ]
  };
  return {
    status: 'display' as const,
    originalResult: result,
    displayResult,
    settings,
    report,
    presentations,
    excludedRows,
    testedCount: table.rowCount - excludedRows.length
  };
}
