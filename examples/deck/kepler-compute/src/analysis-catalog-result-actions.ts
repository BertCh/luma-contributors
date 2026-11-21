// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getCatalogBinding,
  makeCatalogTableInput,
  serializeCatalogTableInput,
  makeWorkflowFromCatalogHistory,
  serializeCatalogInvocation,
  type CatalogDataInput,
  type WorkflowHistoryInputReference,
  type WorkflowOperationRegistry
} from '@luma-contributors/kepler';
import {
  makeAnalysisResultWorkflow,
  createAnalysisResultNextDraft,
  type AnalysisResultCardModel,
  type AnalysisResultNextAction
} from './analysis-result-card-model';

export function createAnalysisCatalogNextDraft(
  model: AnalysisResultCardModel,
  action: AnalysisResultNextAction
) {
  const table = model.result.tables.find(output => output.id === action.tableId);
  if (!table || table.geometryColumn || table.raster)
    return createAnalysisResultNextDraft(model, action);
  if (action.applicability.status === 'blocked')
    throw new Error('Analysis result card: this next analysis is blocked.');
  const input = makeCatalogTableInput(table, {maximumRows: table.rowCount});
  return {
    operationId: action.bindingId,
    drafts: Object.fromEntries(
      Object.entries(action.applicability.inputAssignments).flatMap(([name, datasetId]) =>
        datasetId === action.tableId
          ? [
              [
                `${action.bindingId}:${name}`,
                {
                  mode: 'text' as const,
                  text: serializeCatalogTableInput(input),
                  datasetId: '',
                  field: ''
                }
              ]
            ]
          : []
      )
    ),
    overrides: Object.fromEntries(
      Object.entries(action.applicability.fieldBindings).map(([name, value]) => [
        name,
        typeof value === 'string' ? value : value.join(',')
      ])
    )
  };
}

export function makeAnalysisCatalogResultWorkflow(
  model: AnalysisResultCardModel,
  inputs: Record<string, CatalogDataInput>,
  references: Record<string, WorkflowHistoryInputReference>,
  registry: WorkflowOperationRegistry
) {
  const features = Object.fromEntries(
    Object.entries(inputs).flatMap(([name, input]) =>
      input.type === 'FeatureCollection' ? [[name, input]] : []
    )
  );
  if (Object.keys(features).length === Object.keys(inputs).length)
    return makeAnalysisResultWorkflow(model, features, registry);
  const snapshots = Object.fromEntries(
    Object.entries(inputs).map(([name, input]) => [
      name,
      input.type === 'table'
        ? {mode: 'text' as const, text: serializeCatalogTableInput(input)}
        : references[name]
    ])
  );
  const binding = getCatalogBinding(model.result.operationId)!;
  return makeWorkflowFromCatalogHistory(
    [
      {
        id: 'result',
        operationId: binding.id,
        invocation: serializeCatalogInvocation(binding, model.result.parameters, inputs),
        inputs: snapshots,
        engine: model.result.engine === 'gpu' ? 'auto' : 'cpu'
      }
    ],
    {registry, title: model.title}
  );
}
