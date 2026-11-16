// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  buildKeplerCuratedAnalysisRequest,
  createCuratedWorkflowAnalysisTemplate,
  createWorkflowAnalysisTemplateFromDocument,
  createWorkflowDocument,
  getCatalogBinding,
  getKeplerCuratedAnalysis,
  getWorkflowOperationRegistry,
  replayKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  planWorkflow,
  WORKFLOW_LIMITS,
  type CatalogParameterValue,
  type KeplerCuratedAnalysis,
  type KeplerCuratedAnalysisMapping,
  type KeplerCuratedAnalysisMappingChange,
  type KeplerDatasetProfile,
  type WorkflowAnalysisTemplate,
  type WorkflowDocument,
  type WorkflowStep
} from '@luma-contributors/kepler';
import type {SandboxSampleId} from '../sandbox-samples';
import {getWorkbenchParameterRows} from './workbench-parameter-model';
import type {WorkbenchStore} from './workbench-store';
import {
  connectWorkbenchAnalysisSampleSources,
  getWorkbenchAnalysisSampleData,
  getWorkbenchCuratedSampleId,
  getWorkbenchCuratedSampleParameters,
  getWorkbenchSandboxSampleId,
  isWorkbenchAnalysisSample,
  prepareWorkbenchCuratedSample
} from './workbench-analysis-samples';

function bindParameters(
  parameters: Readonly<Record<string, CatalogParameterValue>>
): WorkflowStep['parameters'] {
  return Object.fromEntries(
    Object.entries(parameters).map(([name, value]) => [name, {from: 'static', value}])
  );
}

export function getWorkbenchAnalysisProfiles(store: WorkbenchStore): KeplerDatasetProfile[] {
  return store.host.listDatasets().flatMap(dataset => {
    const profile =
      !isWorkbenchAnalysisSample(dataset.id) && store.host.getAnalysisProfile?.(dataset.id);
    return profile ? [profile] : [];
  });
}

export function getWorkbenchMappingParameters(
  mapping: KeplerCuratedAnalysisMapping
): Record<string, CatalogParameterValue> {
  return Object.fromEntries(
    (mapping.explicitParameterNames ?? []).map(name => [name, mapping.parameterValues[name]])
  );
}

/** Confirmations describe a dataset. Changing a source must not carry its coordinate/time units. */
export function updateWorkbenchAnalysisMapping(
  mapping: KeplerCuratedAnalysisMapping,
  change: KeplerCuratedAnalysisMappingChange,
  analysis: KeplerCuratedAnalysis,
  profiles: readonly KeplerDatasetProfile[]
): KeplerCuratedAnalysisMapping {
  const changed = updateKeplerCuratedAnalysisMapping(mapping, change, analysis, profiles);
  const parameters = getWorkbenchMappingParameters(mapping);
  if (
    'datasetId' in change &&
    mapping.slots.find(slot => slot.slot === change.slot)?.datasetId !== change.datasetId
  ) {
    delete parameters['coordinateSystem'];
    delete parameters['timestampEncoding'];
  }
  return replayKeplerCuratedAnalysisMapping(analysis, changed.choices, profiles, {parameters});
}

export function getWorkbenchMappingParameterRows(
  analysis: KeplerCuratedAnalysis,
  mapping: KeplerCuratedAnalysisMapping,
  profiles: readonly KeplerDatasetProfile[],
  executionEngine: 'gpu' | 'cpu' = 'gpu'
) {
  if (analysis.execution.status !== 'available') return [];
  const execution = analysis.execution;
  const template = createCuratedWorkflowAnalysisTemplate(analysis);
  const operation = getWorkflowOperationRegistry().get(execution.bindingId);
  if (!operation) return [];
  const parameters = mapping.complete
    ? buildKeplerCuratedAnalysisRequest(analysis, mapping).parameters
    : mapping.parameterValues;
  const step = {
    ...template.document.steps.find(step => step.id === 'analysis')!,
    parameters: bindParameters(parameters)
  };
  const document = {...template.document, inputs: []};
  const profilesByInput = Object.fromEntries(
    mapping.slots.map(slot => [
      execution.inputs[slot.slot],
      profiles.find(profile => profile.datasetId === slot.datasetId)
    ])
  );
  return getWorkbenchParameterRows(document, step, operation, {
    profilesByInput,
    executionEngine
  }).filter(
    row =>
      row.spec.name === 'coordinateSystem' || analysis.exposedParameters.includes(row.spec.name)
  );
}

/** Replace symbolic template sources with the same host dataset adapter used by hand-built steps. */
export function createWorkbenchAnalysisDocument(
  template: WorkflowAnalysisTemplate,
  store: WorkbenchStore,
  mapping?: KeplerCuratedAnalysisMapping
): WorkflowDocument {
  const analysis =
    template.curatedAnalysisId && getKeplerCuratedAnalysis(template.curatedAnalysisId);
  validateRequirement(
    'WorkbenchAnalysisLaunch',
    Boolean(analysis && analysis.execution.status === 'available'),
    'template must name an available curated analysis'
  );
  if (!analysis || analysis.execution.status !== 'available')
    throw new Error('WorkbenchAnalysisLaunch: unavailable template');
  const execution = analysis.execution;
  connectWorkbenchAnalysisSampleSources(store);
  let parameters: Record<string, CatalogParameterValue>;
  let profiles: readonly KeplerDatasetProfile[];
  let datasets: Readonly<Record<string, string>>;
  if (mapping) {
    profiles = getWorkbenchAnalysisProfiles(store);
    const current = replayKeplerCuratedAnalysisMapping(analysis, mapping.choices, profiles, {
      parameters: getWorkbenchMappingParameters(mapping)
    });
    validateRequirement(
      'WorkbenchAnalysisLaunch',
      current.ready,
      'mapping must be ready before running'
    );
    const request = buildKeplerCuratedAnalysisRequest(analysis, current);
    parameters = request.parameters;
    datasets = Object.fromEntries(current.slots.map(slot => [slot.slot, slot.datasetId!]));
  } else {
    prepareWorkbenchCuratedSample(analysis.id);
    parameters = getWorkbenchCuratedSampleParameters(analysis.id);
    datasets = Object.fromEntries(
      analysis.slots.map(slot => [slot.name, getWorkbenchCuratedSampleId(analysis.id, slot.name)])
    );
    profiles = Object.values(datasets).map(
      datasetId => getWorkbenchAnalysisSampleData(datasetId)!.analysisProfile!
    );
  }
  const operation = store.registry.get(execution.bindingId);
  validateRequirement('WorkbenchAnalysisLaunch', Boolean(operation), 'binding must be registered');
  return {
    ...template.document,
    inputs: [],
    settings: {...template.document.settings, autoRun: false},
    steps: template.document.steps.map(step => {
      const slot = template.slots.find(slot => slot.source.stepId === step.id);
      if (!slot)
        return {
          ...step,
          curatedAnalysisId: analysis.id,
          parameters: bindParameters(parameters),
          display: operation!.outputs
            .filter(output => output.kind !== 'table' && output.kind !== 'raster')
            .map(output => output.name)
        };
      const datasetId = datasets[slot.name];
      const profile = profiles.find(profile => profile.datasetId === datasetId)!;
      validateRequirement('WorkbenchAnalysisLaunch', Boolean(profile), 'dataset must still exist');
      const geometry = profile.geometryColumns[0];
      const pointColumns = profile.pointColumns;
      return {
        ...step,
        operationId: 'source-dataset',
        parameters: bindParameters({
          dataset: datasetId,
          datasetLabel: profile.label,
          geometryMode: geometry ? 'geojson-field' : pointColumns ? 'point-columns' : 'none',
          ...(geometry
            ? {geometryField: geometry}
            : pointColumns
              ? {longitudeField: pointColumns.longitude, latitudeField: pointColumns.latitude}
              : {}),
          population: 'all',
          maximumRows: WORKFLOW_LIMITS.maximumSourceRows
        })
      };
    })
  };
}

function getDocumentIssues(document: WorkflowDocument, store: WorkbenchStore): string[] {
  const plan = planWorkflow(document, {registry: store.registry, host: store.host});
  return [...plan.issues, ...Object.values(plan.steps).flatMap(step => step.issues)]
    .filter(issue => issue.severity === 'error')
    .map(issue => issue.message);
}

export function getWorkbenchAnalysisMappingIssues(
  template: WorkflowAnalysisTemplate,
  store: WorkbenchStore,
  mapping: KeplerCuratedAnalysisMapping
): string[] {
  if (!mapping.ready) return [];
  try {
    return getDocumentIssues(createWorkbenchAnalysisDocument(template, store, mapping), store);
  } catch (error) {
    return [error instanceof Error ? error.message : String(error)];
  }
}

export async function runWorkbenchAnalysisDocument(
  document: WorkflowDocument,
  store: WorkbenchStore
) {
  const issues = getDocumentIssues(document, store);
  validateRequirement('WorkbenchAnalysisLaunch', issues.length === 0, issues.join('\n'));
  const workflowId = store.addWorkflow(document);
  const checkActive = () =>
    validateRequirement(
      'WorkbenchAnalysisLaunch',
      store.getState().activeWorkflowId === workflowId,
      'active workflow changed before launch'
    );
  checkActive();
  store.setDock({open: true, inspectorTab: 'result'});
  store.select(['analysis']);
  store.setDialog(null);
  checkActive();
  const record = await store.run();
  validateRequirement(
    'WorkbenchAnalysisLaunch',
    record?.status === 'done',
    record?.steps.flatMap(step => (step.error ? [step.error.message] : [])).join('\n') ||
      'run must finish successfully'
  );
  return record!;
}

export async function launchWorkbenchAnalysis(
  template: WorkflowAnalysisTemplate,
  store: WorkbenchStore,
  mapping?: KeplerCuratedAnalysisMapping
) {
  return runWorkbenchAnalysisDocument(
    createWorkbenchAnalysisDocument(template, store, mapping),
    store
  );
}

const SANDBOX_PAIRINGS: Record<
  SandboxSampleId,
  {
    analysisId?: string;
    operationId?: string;
    title?: string;
    parameters?: Record<string, CatalogParameterValue>;
  }
> = {
  'points-time': {analysisId: 'density-surface', parameters: {cellSize: 500}},
  districts: {
    analysisId: 'excess-risk',
    parameters: {casesProperty: 'events', populationProperty: 'population'}
  },
  connections: {operationId: 'geometry-measures', title: 'How long are the connections?'},
  trips: {operationId: 'geometry-measures', title: 'How long are the timed routes?'},
  'trip-observations': {
    analysisId: 'track-quality',
    parameters: {
      trackProperty: 'vehicle_id',
      timeProperty: 'observed_at',
      timestampEncoding: 'unix-millisecond'
    }
  },
  'h3-cells': {
    operationId: 'geometry-measures',
    title: 'What are the areas and perimeters of these H3 cells?'
  },
  'numeric-grid': {analysisId: 'idw-surface', parameters: {valueProperty: 'value', cellSize: 500}},
  'directed-network': {
    operationId: 'network-structure',
    title: 'Where are the junctions and dead ends in this network?'
  }
};

export function createWorkbenchSandboxAnalysis(
  sampleId: SandboxSampleId
): WorkflowAnalysisTemplate {
  const pairing = SANDBOX_PAIRINGS[sampleId];
  const analysis = pairing.analysisId ? getKeplerCuratedAnalysis(pairing.analysisId) : undefined;
  const execution = analysis?.execution.status === 'available' ? analysis.execution : undefined;
  const operationId = execution?.bindingId ?? pairing.operationId!;
  const binding = getCatalogBinding(operationId)!;
  const datasetId = getWorkbenchSandboxSampleId(sampleId);
  const data = getWorkbenchAnalysisSampleData(datasetId)!;
  const parameters = {
    ...execution?.parameters,
    coordinateSystem: 'EPSG:4326',
    ...pairing.parameters
  };
  const document = createWorkflowDocument({
    id: `sandbox-${sampleId}`,
    title: analysis?.title ?? pairing.title
  });
  document.steps = [
    {
      id: 'source',
      operationId: 'source-dataset',
      inputs: {},
      parameters: bindParameters({
        dataset: datasetId,
        geometryField: 'geometry',
        population: 'all',
        maximumRows: WORKFLOW_LIMITS.maximumSourceRows
      }),
      position: {x: 0, y: 0}
    },
    {
      id: 'analysis',
      operationId,
      ...(analysis ? {curatedAnalysisId: analysis.id} : {}),
      inputs: {[binding.inputs[0].name]: {stepId: 'source', output: 'data'}},
      parameters: bindParameters(parameters),
      position: {x: 250, y: 0},
      display: binding.outputs
        .filter(output => output.geometry !== 'none')
        .map(output => output.tableId)
    }
  ];
  validateRequirement(
    'WorkbenchSandboxAnalysis',
    data.table.rowCount > 0,
    'sample must contain rows'
  );
  const template = createWorkflowAnalysisTemplateFromDocument({
    id: document.id,
    title: document.title,
    summary: analysis?.method ?? binding.summary,
    document
  });
  return {
    ...template,
    sample: {
      id: sampleId,
      license: 'MIT',
      attribution: 'vis.gl contributors',
      reference: 'examples/deck/kepler-compute/src/sandbox-samples.ts'
    }
  };
}

export async function launchWorkbenchSandboxAnalysis(
  store: WorkbenchStore,
  sampleId: SandboxSampleId
) {
  connectWorkbenchAnalysisSampleSources(store);
  return runWorkbenchAnalysisDocument(createWorkbenchSandboxAnalysis(sampleId).document, store);
}
