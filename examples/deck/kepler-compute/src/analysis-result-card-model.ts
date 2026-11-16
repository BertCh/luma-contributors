// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  KEPLER_CATALOG_BINDINGS,
  addWorkflowStep,
  describeKeplerCatalogResult,
  evaluateKeplerOperationApplicability,
  getCatalogBinding,
  getKeplerBindingResultPresentations,
  getWorkflowFeatures,
  makeWorkflowData,
  makeWorkflowFromCatalogHistory,
  parseWorkflowDocument,
  profileKeplerWorkbenchData,
  profileKeplerCatalogOutput,
  serializeKeplerDatasetProfileDeclarations,
  resolveKeplerOperationDescriptor,
  serializeCatalogInvocation,
  type CatalogFeatureCollection,
  type CatalogRunResult,
  type KeplerOperationApplicability,
  type KeplerOperationDescriptor,
  type KeplerResultReport,
  type KeplerResultReportContext,
  type KeplerDatasetProfileDeclarations,
  type WorkflowOperationRegistry
} from '@luma-contributors/kepler';
import type {WorkflowDocument} from '@luma-contributors/kepler';
import type {WorkbenchStore} from './workflow/workbench-store';

export type AnalysisResultStackEntry = {workflowId: string; stepIds: readonly string[]};

/** Membership follows the saved destination, including undo, redo, removal and workflow deletion. */
export function hasAnalysisResultStackEntry(
  entry: AnalysisResultStackEntry,
  workflows: readonly WorkflowDocument[]
) {
  const workflow = workflows.find(document => document.id === entry.workflowId);
  return Boolean(
    workflow &&
      entry.stepIds.length > 0 &&
      entry.stepIds.every(stepId => workflow.steps.some(step => step.id === stepId))
  );
}

/** Keep using the existing workbench edits, while retaining the identities they actually insert. */
export function keepAnalysisResultWorkflow(
  document: WorkflowDocument,
  store: WorkbenchStore,
  save: boolean
): AnalysisResultStackEntry {
  const previous = store.getState().document;
  let workflowId: string;
  let stepIds: string[];
  if (save || !previous) {
    workflowId = store.addWorkflow(document);
    if (!workflowId)
      throw new Error(store.getState().notice?.message ?? 'Could not add this run to the stack.');
    stepIds = store
      .getState()
      .workflows.find(workflow => workflow.id === workflowId)!
      .steps.map(step => step.id);
    if (save && !store.saveTemplate(workflowId))
      throw new Error(store.getState().notice?.message ?? 'Could not save this workflow.');
  } else {
    workflowId = previous.id;
    if (!store.insertWorkflow(document))
      throw new Error(store.getState().notice?.message ?? 'Could not add this run to the stack.');
    const existingIds = new Set(previous.steps.map(step => step.id));
    stepIds = store
      .getState()
      .workflows.find(workflow => workflow.id === workflowId)!
      .steps.filter(step => !existingIds.has(step.id))
      .map(step => step.id);
  }
  return {workflowId, stepIds};
}

export type AnalysisResultNextAction = {
  bindingId: string;
  label: string;
  question: string;
  tableId: string;
  applicability: KeplerOperationApplicability;
};

export type AnalysisResultCardModel = ReturnType<typeof createAnalysisResultCardModel>;

/** Immutable run evidence; callers capture this when execution finishes, never from a later form. */
export function createAnalysisResultCardModel(
  result: CatalogRunResult,
  context: KeplerResultReportContext = {},
  options: {
    descriptor?: KeplerOperationDescriptor;
    report?: KeplerResultReport;
    executionEngine?: 'cpu' | 'gpu';
    outputProfiles?: readonly ReturnType<typeof profileKeplerWorkbenchData>[];
    /** The catalog's input editor reads features; the workbench also accepts typed tables. */
    featureInputsOnly?: boolean;
  } = {}
) {
  if (result.sensitivity) context = {...context, parameterSearch: true};
  const binding = getCatalogBinding(result.operationId);
  if (!binding) throw new Error(`Analysis result card: unknown binding ${result.operationId}`);
  const descriptor = options.descriptor ?? resolveKeplerOperationDescriptor(binding).descriptor;
  const report =
    options.report ?? describeKeplerCatalogResult(result, binding, descriptor, context);
  const presentations = getKeplerBindingResultPresentations(result, descriptor);
  const profiles =
    options.outputProfiles ??
    result.tables.map(table =>
      profileKeplerCatalogOutput(
        makeWorkflowData(table),
        result,
        context.profiles ?? {},
        descriptor
      )
    );
  const next = getAnalysisResultNextActions(
    report,
    profiles,
    options.executionEngine ?? 'cpu',
    options.featureInputsOnly
      ? KEPLER_CATALOG_BINDINGS.filter(candidate =>
          candidate.inputs.every(input => !input.dataKind || input.dataKind === 'features')
        )
      : KEPLER_CATALOG_BINDINGS
  );
  return {
    title: binding.title,
    result,
    context,
    report,
    presentations,
    outputProfiles: profiles,
    next,
    // Keep the original values and provenance; formatted sentences alone cannot reproduce a run.
    json: JSON.stringify({schemaVersion: 1, result, context, report}, null, 2)
  };
}

/** Follow-on questions are checked against actual output rows, including table-only results. */
export function getAnalysisResultNextActions(
  report: KeplerResultReport,
  profiles: readonly ReturnType<typeof profileKeplerWorkbenchData>[],
  executionEngine: 'cpu' | 'gpu',
  bindings = KEPLER_CATALOG_BINDINGS
): AnalysisResultNextAction[] {
  const statusOrder = {ready: 0, 'needs-mapping': 1, 'needs-confirmation': 2, blocked: 3};
  return report.next.slice(0, 3).flatMap(question => {
    const candidates: AnalysisResultNextAction[] = [];
    for (const binding of bindings) {
      const {descriptor} = resolveKeplerOperationDescriptor(binding);
      if (!descriptor.questions.includes(question.question)) continue;
      for (const profile of profiles) {
        const applicability = evaluateKeplerOperationApplicability(binding, descriptor, [profile], {
          requiredDatasetId: profile.datasetId,
          executionEngine
        });
        candidates.push({
          bindingId: binding.id,
          label: binding.title,
          question: question.label,
          tableId: profile.datasetId,
          applicability
        });
      }
    }
    candidates.sort(
      (first, second) =>
        statusOrder[first.applicability.status] - statusOrder[second.applicability.status] ||
        first.bindingId.localeCompare(second.bindingId) ||
        first.tableId.localeCompare(second.tableId)
    );
    return candidates.slice(0, 1);
  });
}

/** Open the exact compatible output in the catalog; unresolved mappings remain visible there. */
export function createAnalysisResultNextDraft(
  model: AnalysisResultCardModel,
  action: AnalysisResultNextAction
) {
  if (action.applicability.status === 'blocked')
    throw new Error('Analysis result card: this next analysis is blocked.');
  const table = model.result.tables.find(output => output.id === action.tableId);
  if (!table)
    throw new Error('Analysis result card: the next analysis needs an output from this run.');
  const text = JSON.stringify(getWorkflowFeatures(makeWorkflowData(table)));
  const profile = model.outputProfiles.find(output => output.datasetId === action.tableId);
  const coordinateModel =
    profile?.coordinates?.source === 'declared' ? profile.coordinates.model : undefined;
  const profileDeclarations = coordinateModel
    ? serializeKeplerDatasetProfileDeclarations({coordinateModel})
    : undefined;
  const drafts = Object.fromEntries(
    Object.entries(action.applicability.inputAssignments).flatMap(([name, datasetId]) =>
      datasetId === action.tableId
        ? [
            [
              `${action.bindingId}:${name}`,
              {
                mode: 'text' as const,
                text,
                datasetId: '',
                field: '',
                ...(profileDeclarations ? {profileDeclarations} : {})
              }
            ]
          ]
        : []
    )
  );
  const binding = getCatalogBinding(action.bindingId)!;
  const coordinateSystem =
    coordinateModel === 'planar'
      ? 'planar'
      : coordinateModel === 'geographic'
        ? 'EPSG:4326'
        : undefined;
  return {
    operationId: action.bindingId,
    drafts,
    overrides: {
      ...getAnalysisResultNextFields(action),
      ...(binding.parameters.some(parameter => parameter.name === 'coordinateSystem') &&
      (coordinateSystem === 'planar' || coordinateSystem === 'EPSG:4326')
        ? {coordinateSystem}
        : {})
    }
  };
}

function getAnalysisResultNextFields(action: AnalysisResultNextAction) {
  return Object.fromEntries(
    Object.entries(action.applicability.fieldBindings).map(([name, value]) => [
      name,
      typeof value === 'string' ? value : value.join(',')
    ])
  );
}

/** Use the existing workflow connection validator and retain the proposed field-role assignment. */
export function addAnalysisResultNextStep(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  action: AnalysisResultNextAction,
  source: {stepId: string; output: string}
) {
  const input = Object.entries(action.applicability.inputAssignments).find(
    ([, datasetId]) => datasetId === action.tableId
  )?.[0];
  if (!input || action.applicability.status === 'blocked')
    throw new Error('Analysis result card: choose an applicable output before adding a next step.');
  return addWorkflowStep(document, registry, {
    operationId: action.bindingId,
    connectFrom: source,
    connectTo: input,
    parameters: getAnalysisResultNextFields(action)
  });
}

/** Reuse history promotion and its limits; embed exactly the bounded inputs read by this run. */
export function makeAnalysisResultWorkflow(
  model: AnalysisResultCardModel,
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  registry: WorkflowOperationRegistry
) {
  const binding = getCatalogBinding(model.result.operationId)!;
  const document = makeWorkflowFromCatalogHistory(
    [
      {
        id: 'result',
        operationId: binding.id,
        invocation: serializeCatalogInvocation(binding, model.result.parameters, inputs),
        inputs: Object.fromEntries(
          Object.entries(inputs).map(([name, collection]) => [
            name,
            {
              mode: 'text' as const,
              text: JSON.stringify(collection),
              profileDeclarations: serializeKeplerDatasetProfileDeclarations(
                getAnalysisResultInputDeclarations(model.context.profiles?.[name])
              )
            }
          ])
        ),
        engine: model.result.engine === 'gpu' ? 'auto' : 'cpu'
      }
    ],
    {registry, title: model.title}
  );
  // Promotion elides binding constants. A kept run must retain all its executed values even
  // when a later profile or descriptor would derive a different default.
  return parseWorkflowDocument({
    ...document,
    steps: document.steps.map(step => {
      if (step.operationId === binding.id) {
        return {
          ...step,
          ...(model.result.sensitivity ? {sensitivity: model.result.sensitivity} : {}),
          parameters: Object.fromEntries(
            Object.entries(model.result.parameters).map(([name, value]) => [
              name,
              {from: 'static', value}
            ])
          )
        };
      }
      const inputName = Object.entries(
        document.steps.find(candidate => candidate.operationId === binding.id)!.inputs
      ).find(([, connection]) => connection.stepId === step.id)?.[0];
      const coordinates = inputName ? model.context.profiles?.[inputName]?.coordinates : undefined;
      return coordinates?.source === 'declared'
        ? {
            ...step,
            parameters: {
              ...step.parameters,
              coordinateSystem: {
                from: 'static',
                value: coordinates.model === 'geographic' ? 'EPSG:4326' : 'planar'
              }
            }
          }
        : step;
    })
  });
}

function getAnalysisResultInputDeclarations(
  profile: NonNullable<KeplerResultReportContext['profiles']>[string]
): KeplerDatasetProfileDeclarations {
  return {
    ...(profile?.coordinates?.source === 'declared' &&
    (profile.coordinates.model === 'planar' || profile.coordinates.model === 'geographic')
      ? {coordinateModel: profile.coordinates.model}
      : {}),
    ...(profile?.rowSemantics?.source === 'declared'
      ? {rowSemantics: profile.rowSemantics.value}
      : {}),
    measurement: Object.fromEntries(
      (profile?.columns ?? []).flatMap(column =>
        column.measurementSource === 'declared' && column.measurement
          ? [[column.name, column.measurement]]
          : []
      )
    )
  };
}
