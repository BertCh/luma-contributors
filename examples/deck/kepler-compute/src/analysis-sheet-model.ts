// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  makeWorkflowOperationFromCatalogBinding,
  KEPLER_CURATED_ANALYSES,
  getCatalogBinding,
  getKeplerWorkbenchCuratedPresets,
  type CatalogOperationBinding,
  type CatalogParameterValue,
  type WorkflowParameterSpec,
  type KeplerDatasetProfile
} from '@luma-contributors/kepler';
import {evaluateGPUAnalysisParameterCondition} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  createAnalysisCatalogFormModel,
  describeAnalysisCatalogParameter
} from './analysis-catalog-form-model';
import type {ResidentAnalysisSheetField} from './analysis-sheet';

/** Resident recipe adapters share field rendering without substituting a catalog method. */
export function createResidentSheetFields(
  values: Readonly<Record<string, CatalogParameterValue>>,
  specifications: readonly (WorkflowParameterSpec & {controlId: string})[],
  source: (name: string) => string,
  columns: (name: string) => readonly string[]
): ResidentAnalysisSheetField[] {
  return specifications.map(({controlId, ...specification}) => ({
    specification,
    controlId,
    value: values[specification.name] ?? '',
    source: source(specification.name),
    columns: specification.fieldRole ? columns(specification.name) : undefined
  }));
}

/** Only selected field names and bounded settings are checked; source rows stay resident. */
export function getResidentSheetChecks(
  fields: readonly ResidentAnalysisSheetField[],
  readSettings: () => unknown
): string[] {
  const failures = fields.flatMap(field =>
    field.specification.fieldRole && !field.columns?.includes(String(field.value))
      ? [`${field.specification.label}: choose an available prepared field.`]
      : []
  );
  try {
    readSettings();
  } catch (failure) {
    failures.push(
      `${failure instanceof Error ? failure.message : String(failure)}. Review the settings before applying.`
    );
  }
  return failures;
}

/** Empty drafts must not silently become numeric zero at the recipe boundary. */
export function readResidentSheetNumber(
  values: Readonly<Record<string, CatalogParameterValue>>,
  name: string
): number {
  const value = values[name];
  if ((typeof value !== 'number' && typeof value !== 'string') || String(value).trim() === '')
    throw new Error(`Declare ${name}`);
  return Number(value);
}

export type AnalysisSheetForm = ReturnType<typeof createAnalysisCatalogFormModel>;

/** Use the same field-role/control adapter and resolved conditions as the workbench. */
export function createAnalysisSheetModel(
  binding: CatalogOperationBinding,
  form: AnalysisSheetForm,
  profiles: Readonly<Record<string, KeplerDatasetProfile | undefined>>
) {
  const operation = makeWorkflowOperationFromCatalogBinding(binding);
  const rows = operation.parameters.flatMap(specification => {
    if (form.resolved.inactive.includes(specification.name)) return [];
    const resolution = form.resolved.resolutions.find(
      candidate => candidate.parameter === specification.name
    );
    const profile = specification.columnsFrom ? profiles[specification.columnsFrom] : undefined;
    const field = specification.fieldRole;
    return [
      {
        specification,
        value: form.resolved.parameters[specification.name] ?? '',
        source: resolution ? describeAnalysisCatalogParameter(resolution) : 'Declared constant',
        columns: field
          ? (profile?.columns ?? [])
              .filter(
                column =>
                  field.accepts.includes(column.type) ||
                  (column.type === 'integer' && field.accepts.includes('number'))
              )
              .map(column => column.name)
          : [],
        input: field ? specification.columnsFrom : undefined,
        required: field
          ? typeof field.required === 'boolean'
            ? field.required
            : evaluateGPUAnalysisParameterCondition(field.required, form.resolved.parameters)
          : false,
        apply: specification.apply ?? 'rebuild'
      }
    ];
  });
  return {
    rows,
    basic: rows.filter(row => !row.input && row.specification.level !== 'advanced'),
    advanced: rows.filter(row => !row.input && row.specification.level === 'advanced'),
    methodDifference: describeAnalysisSheetMethod(binding.id)
  };
}

export type AnalysisSheetRow = ReturnType<typeof createAnalysisSheetModel>['rows'][number];

/** Keep statistically different entry points explicit while handwritten panels migrate. */
export function describeAnalysisSheetMethod(bindingId: string): string | undefined {
  if (bindingId === 'local-moran-statistics')
    return 'Catalog Local Moran uses row-standardized weights, 999 permutations and Benjamini–Hochberg correction by default. The resident Local Moran panel uses binary weights and defaults to analytic conditional-normal inference; these are different methods.';
  if (bindingId === 'kriging-grid')
    return 'Catalog kriging uses the supplied variogram without curve fitting or cross-validation. The resident Kriging panel offers variogram fitting and cross-validation. The curated kriging analysis includes separate diagnostics.';
  return undefined;
}

/** Full selected inputs only: a profile sample used for defaults is not a sampled run. */
export const ANALYSIS_SHEET_PREVIEW_POLICY = {
  scope: 'full-population',
  description:
    'Full population of each selected input. No viewport or row sampling is applied to the calculation.',
  inference:
    'Sampled or viewport previews must suppress significance classes. Keeping an inferential result requires a full-population computation.'
} as const;

/** Live edits coalesce within a frame; rebuild controls commit on release before scheduling. */
export function getAnalysisSheetPreviewDelay(apply: 'live' | 'rebuild'): number {
  return apply === 'live' ? 16 : 0;
}

/** Qualification inventory comes from the real curated registry, never an optimistic fixed list. */
export function getAnalysisSheetQualificationCases() {
  const presets = getKeplerWorkbenchCuratedPresets();
  return KEPLER_CURATED_ANALYSES.filter(analysis => analysis.wave === 1).map(analysis => {
    const execution = analysis.execution;
    const binding =
      execution.status === 'available' ? getCatalogBinding(execution.bindingId) : undefined;
    const operation = binding ? makeWorkflowOperationFromCatalogBinding(binding) : undefined;
    const preset = presets.find(candidate => candidate.curatedAnalysisId === analysis.id);
    const live = operation?.parameters.filter(parameter => parameter.apply === 'live') ?? [];
    return {
      templateId: analysis.id,
      sampleId: analysis.sampleId,
      bindingId: binding?.id,
      engine: binding?.engine,
      parameters: preset?.parameters,
      inputs: preset?.sampleGeoJSONByInput,
      inputLabels: binding?.inputs.map(input => ({name: input.name, label: input.label})),
      live,
      unavailable:
        execution.status === 'needs-binding'
          ? execution.reason
          : !binding
            ? 'No registered executable binding'
            : !binding.planPreview
              ? `${binding.engine}: binding has no retained preview executor`
              : !preset?.sampleGeoJSONByInput
                ? 'Sample needs a typed workflow input; no GeoJSON sheet path'
                : live.length === 0
                  ? 'No live parameter is declared'
                  : undefined
    };
  });
}
