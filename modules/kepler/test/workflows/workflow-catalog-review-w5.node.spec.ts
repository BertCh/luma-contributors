// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import type {Device} from '@luma.gl/core';
import {createRasterHandoffFixture} from './workflow-raster-handoff-fixtures';
import {planWorkflow, resolveWorkflowParameters} from '../../src/workflows/workflow-plan';
import {getWorkflowOperationRegistry} from '../../src/workflows/workflow-operations';
import {createWorkflowDocument} from '../../src/workflows/workflow-document';
import {
  addWorkflowStep,
  connectWorkflowSteps,
  replaceWorkflowStepOperation
} from '../../src/workflows/workflow-edit';
import {
  profileAnalysisCatalogInput,
  createAnalysisCatalogFormModel,
  resolveAnalysisCatalogInputProfiles,
  restoreAnalysisCatalogInvocation,
  editAnalysisCatalogOverride
} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-form-model';
import {
  getCatalogBinding,
  listCatalogEntries,
  normalizeCatalogParameters,
  parseCatalogInvocation,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import {makeCatalogTableInput} from '../../src/analysis/catalog-data-input';
import {listQualifiedCatalogEntries} from '../../src/analysis/catalog-qualified-entries';

import {
  readCatalogFeatureInputs,
  readCatalogTextInput
} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {
  createAnalysisCatalogNextDraft,
  makeAnalysisCatalogResultWorkflow
} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-result-actions';
import {
  AnalysisCatalogBlockedEntries,
  describeCatalogInput,
  describeCatalogCapacity
} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-display';
import {createAnalysisResultCardModel} from '../../../../examples/deck/kepler-compute/src/analysis-result-card-model';
import {readNativeDatasetData} from '../../../../examples/deck/kepler-compute/src/workflow/workbench-host-data';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {resolveKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {makeFixtureHost} from './workflow-fixtures';
const hostRequire = createRequire(
  new URL('../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

import {createCatalogIcpWorkflowInputs} from './workflow-catalog-icp-fixture';
import {
  evaluateKeplerWorkbenchInputs,
  profileKeplerWorkbenchData
} from '../../src/descriptors/workbench-adapters';

import {expectCatalogNumericRows} from './workflow-catalog-numeric-assertions';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {makeValidationResult} from '../../src/analysis/catalog-validation-common';
import {
  createWorkflowOperationRegistry,
  makeWorkflowOperationFromCatalogBinding
} from '../../src/workflows/workflow-operations';
import {
  parseCatalogTableInput,
  serializeCatalogTableInput
} from '../../src/analysis/catalog-data-input';

const line = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: [
          [-1, 1],
          [3, 1]
        ]
      },
      properties: {}
    }
  ]
};

test('B1 first-run surface sources remain raster and unknown representations defer admission', () => {
  const {host, registry, document} = createRasterHandoffFixture({} as Device);
  const planned = planWorkflow(document, {host, registry});
  expect(
    planned.steps['prepare'].issues.filter(issue => issue.code === 'incompatible-port')
  ).toEqual([]);
  const unknown = planWorkflow(document, {registry});
  expect(
    unknown.steps['prepare'].issues.filter(issue => issue.code === 'incompatible-port')
  ).toEqual([]);
});

test('B2 editor and automatic connection inherit clipped lines, including replacement', () => {
  const registry = getWorkflowOperationRegistry();
  const source = addWorkflowStep(createWorkflowDocument({title: 'Clipped line'}), registry, {
    id: 'source',
    operationId: 'source-geojson',
    parameters: {text: JSON.stringify(line), coordinateSystem: 'planar'}
  });
  const clip = addWorkflowStep(source.document, registry, {
    id: 'clip',
    operationId: 'geometry-convex-clip',
    connectFrom: {stepId: source.stepId, output: 'data'},
    connectTo: 'features'
  });
  const target = addWorkflowStep(clip.document, registry, {
    id: 'refine',
    operationId: 'geometry-line-segmentize'
  });
  const connected = connectWorkflowSteps(target.document, registry, {
    from: {stepId: clip.stepId, output: 'clipped'},
    to: {stepId: target.stepId, input: 'lines'}
  });
  expect(connected.steps.find(step => step.id === 'refine')?.inputs['lines']).toEqual({
    stepId: 'clip',
    output: 'clipped'
  });
  const automatic = addWorkflowStep(clip.document, registry, {
    operationId: 'geometry-line-segmentize',
    connectFrom: {stepId: clip.stepId, output: 'clipped'}
  });
  expect(
    automatic.document.steps.find(step => step.id === automatic.stepId)?.inputs['lines']
  ).toEqual({stepId: 'clip', output: 'clipped'});
  const replaced = replaceWorkflowStepOperation(
    connected,
    registry,
    'refine',
    'geometry-line-segmentize',
    {maximumSegmentLength: 2}
  );
  expect(replaced.steps.find(step => step.id === 'refine')?.inputs['lines']).toEqual({
    stepId: 'clip',
    output: 'clipped'
  });
});

test('B6 explicit table profiling disables coordinate inference and keeps table output kinds', () => {
  const input = makeCatalogTableInput(
    {
      id: 'pvalues',
      label: 'P-values',
      rowCount: 2,
      columns: {longitude: [1, 2], latitude: [3, 4], pvalue: [0.03, 0.12]},
      columnTypes: {longitude: 'float64', latitude: 'float64', pvalue: 'float64'}
    },
    {maximumRows: 2}
  );
  const binding = getCatalogBinding('validation-fdr')!;
  const profile = profileAnalysisCatalogInput(binding, 'points', input);
  expect(profile.kind).toBe('attributes');
  const form = createAnalysisCatalogFormModel(binding, {points: profile});
  expect(form.applicability.outputKinds['validation-fdr']).toBe('attributes');
});

test('B9 qualified matches combine declared identities with qualified execution stages', () => {
  const contributor = {package: '@host/analysis', exportName: 'FirstContributor'};
  const stage = {package: '@host/analysis', exportName: 'SecondStage'};
  const entries = listQualifiedCatalogEntries(
    [
      {
        ...getCatalogBinding('geometry-measures')!,
        id: 'union-fixture',
        engine: 'gpu',
        contributorIdentities: [contributor],
        stages: [{identity: stage, execution: 'gpu'}]
      }
    ],
    [],
    []
  );
  expect(entries.find(entry => entry.identity.exportName === stage.exportName)).toMatchObject({
    bindingIds: ['union-fixture'],
    runnable: 'catalog-binding',
    engine: 'gpu'
  });
});

test('B4 sensitivity narrows feature transport and rejects a table with a named problem', async () => {
  await expect(readCatalogFeatureInputs(async () => ({points: tableInput}))).rejects.toThrow(
    'points requires feature input'
  );
  const features = {
    type: 'FeatureCollection' as const,
    features: [{geometry: {type: 'Point' as const, coordinates: [0, 0]}}]
  };
  expect(await readCatalogFeatureInputs(async () => ({points: features}))).toEqual({
    points: features
  });
});

test('B5 an enabled score action carries a typed table through draft creation and loading', () => {
  const target = getCatalogBinding('validation-ensemble-scores')!;
  const profile = resultModel.outputProfiles[0];
  const applicability = evaluateKeplerOperationApplicability(
    target,
    resolveKeplerOperationDescriptor(target).descriptor,
    [profile],
    {
      executionEngine: 'gpu',
      inputAssignments: {points: profile.datasetId},
      overrides: {observationProperty: 'observed', memberProperties: 'member1,member2'}
    }
  );
  expect(applicability.status).toBe('ready');
  const next = createAnalysisCatalogNextDraft(resultModel, {
    bindingId: target.id,
    label: 'Ensemble scores',
    question: 'how-related',
    tableId: 'validation-fdr',
    applicability
  });
  const draft = next.drafts[`${target.id}:points`];
  const loaded = readCatalogTextInput(draft.text, target.inputs[0]);
  expect(loaded.type).toBe('table');
  if (loaded.type !== 'table') throw new Error('Expected table');
  expect(loaded.table.columns['observed']).toEqual([1, 3]);
  const profiles = resolveAnalysisCatalogInputProfiles(target, next.drafts, () => undefined);
  expect(profiles.problems).toEqual([]);
  expect(profiles.profiles['points']?.kind).toBe('attributes');
  const nextForm = createAnalysisCatalogFormModel(target, profiles.profiles, {
    overrides: next.overrides,
    executionEngine: 'gpu'
  });
  expect(nextForm.runDisabled).toBe(false);
});

test('B7 kept native tables replay original reserved-name values into mapped parameters', async () => {
  const binding = getCatalogBinding('validation-fdr')!;
  const model = createAnalysisResultCardModel(
    {
      ...resultModel.result,
      parameters: normalizeCatalogParameters(binding, {pValueProperty: 'row_index'})
    },
    {},
    {executionEngine: 'gpu'}
  );
  const registry = getWorkflowOperationRegistry();
  const document = makeAnalysisCatalogResultWorkflow(
    model,
    {points: tableInput},
    {points: {mode: 'native', datasetId: 'native', field: ''}},
    registry
  );
  const source = document.steps.find(step => step.operationId.startsWith('source-'))!;
  const host = makeFixtureHost([
    {
      id: 'native',
      label: 'Native',
      rowCount: 2,
      fields: [{name: 'row_index', type: 'real'}],
      geometryFields: [],
      resident: false
    }
  ]);
  host.readDataset = async () =>
    readNativeDatasetData(
      {
        id: 'native',
        label: 'Native',
        fields: [{name: 'row_index', type: 'real'}],
        dataContainer: {numRows: () => 2, valueAt: row => [0.03, 0.12][row]}
      },
      {datasetId: 'native', geometry: {mode: 'none'}, population: 'all', maximumRows: 2}
    );
  const operation = registry.get(source.operationId)!;
  const data = await operation.run(
    {
      device: null,
      host,
      engine: 'cpu',
      workflowId: 'replay',
      stepId: source.id,
      signal: new AbortController().signal
    },
    {},
    operation.normalizeParameters(resolveWorkflowParameters(document, source, operation))
  );
  const target = document.steps.find(step => step.operationId === binding.id)!;
  const parameter = target.parameters['pValueProperty'];
  if (parameter.from !== 'static') throw new Error('Expected static field mapping');
  expect(data.outputs['data'].table.columns[String(parameter.value)]).toEqual([0.03, 0.12]);
  let dispatchedValues: readonly (string | number | boolean | null)[] = [];
  const replayOperation = makeWorkflowOperationFromCatalogBinding({
    ...binding,
    runDataGpu: async (_context, inputs, values) => {
      const input = inputs['points'];
      if (input.type !== 'table') throw new Error('Expected dispatcher table');
      dispatchedValues = input.table.columns[String(values['pValueProperty'])];
      return makeValidationResult(
        binding.id,
        binding.title,
        input,
        {adjusted_pvalue: [0.06, 0.24], significant: [0, 0]},
        {},
        []
      );
    }
  });
  const replayRegistry = createWorkflowOperationRegistry([
    ...registry.list().filter(candidate => candidate.id !== binding.id),
    replayOperation
  ]);
  const replayDevice = {} as Device;
  host.getDevice = () => replayDevice;
  const runner = new WorkflowRunner({host, registry: replayRegistry});
  try {
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.status).toBe('done');
    expect(dispatchedValues).toEqual([0.03, 0.12]);
  } finally {
    await runner.destroy();
  }
});

test('B8 the blocked panel reaches both visibility identities, including the last row', () => {
  const entries = listCatalogEntries({lensScenarios: []}).filter(entry => entry.runnable === 'no');
  const markup = renderToStaticMarkup(createElement(AnalysisCatalogBlockedEntries, {entries}));
  expect((markup.match(/<li(?:\s|>)/g) ?? []).length).toBe(entries.length);
  expect(markup).toContain('addVisibilityGraphRecipe · @luma-contributors/analysis/gpu-network');
  expect(markup).toContain(
    'addVisibilityGraphRecipe · @luma-contributors/analysis/gpu-spatial-analysis'
  );
  const visibleText = markup.replace(/<[^>]*>/g, '');
  expect([...visibleText.matchAll(/No executable host binding/g)]).toHaveLength(1);
});

test('B10 a serialized retired parameter is rejected and can be removed', async () => {
  const binding = getCatalogBinding('pointcloud-icp')!;
  const text = JSON.stringify({
    schemaVersion: 1,
    operationId: binding.id,
    parameters: {maximumIterations: 13, iterations: 18},
    inputs: {}
  });
  expect(() => parseCatalogInvocation(text)).toThrow(
    /iterations was removed; use maximumIterations/
  );
  const restored = restoreAnalysisCatalogInvocation(text);
  await expect(
    runCatalogOperationAsync(binding, restored.invocation.parameters, {})
  ).rejects.toThrow(/iterations was removed; use maximumIterations/);
  const blocked = createAnalysisCatalogFormModel(
    restored.binding,
    {},
    {overrides: restored.invocation.parameters}
  );
  expect(blocked.runDisabled).toBe(true);
  expect(blocked.problems.join(' ')).toMatch(/iterations was removed; use maximumIterations/);
  const corrected = editAnalysisCatalogOverride(
    binding,
    restored.invocation.parameters,
    'iterations',
    undefined
  );
  const readyParameters = normalizeCatalogParameters(binding, corrected);
  expect(readyParameters['maximumIterations']).toBe(13);
  expect(readyParameters).not.toHaveProperty('iterations');
});

test('S2 table selection describes its row capacity without a geometry vertex limit', () => {
  const input = getCatalogBinding('validation-fdr')!.inputs[0];
  expect(describeCatalogInput(input, true)).toContain('table rows');
  expect(describeCatalogInput(input, true)).toContain('2,048');
  expect(describeCatalogInput(input, true)).not.toContain('vertices');
  expect(describeCatalogInput(input, false)).toContain('table');
  const pending = describeCatalogCapacity(
    {
      label: input.label,
      maximumRows: input.maximumFeatures,
      maximumVertices: input.maximumVertices
    },
    true
  );
  expect(pending).toContain('table rows');
  expect(pending).not.toContain('vertices');
});

const tableInput = makeCatalogTableInput(
  {
    id: 'validation-fdr',
    label: 'FDR output',
    rowCount: 2,
    columns: {
      row_index: [0.03, 0.12],
      observed: [1, 3],
      predicted: [2, 3],
      variance: [0, 0],
      member1: [2, 3],
      member2: [2, 3],
      pvalue: [0.01, 0.2]
    },
    columnTypes: {
      row_index: 'float64',
      observed: 'float64',
      predicted: 'float64',
      variance: 'float64',
      member1: 'float64',
      member2: 'float64',
      pvalue: 'float64'
    }
  },
  {maximumRows: 2}
);
const resultModel = createAnalysisResultCardModel(
  {
    operationId: 'validation-fdr',
    engine: 'gpu',
    parameters: normalizeCatalogParameters(getCatalogBinding('validation-fdr')!, {}),
    tables: [tableInput.table],
    diagnostics: [],
    summary: {},
    dispatchedContributors: []
  },
  {},
  {executionEngine: 'gpu'}
);

test('B3 the queued ICP fixture passes real distinct-row admission before GPU execution', () => {
  const inputs = createCatalogIcpWorkflowInputs();
  const admission = evaluateKeplerWorkbenchInputs(
    'pointcloud-icp',
    {
      source: profileKeplerWorkbenchData(inputs.source),
      target: profileKeplerWorkbenchData(inputs.target)
    },
    {
      overrides: {coordinateSystem: 'planar', maximumDistance: 2, maximumIterations: 8},
      executionEngine: 'gpu'
    }
  );
  expect(admission.status, JSON.stringify(admission.missing)).toBe('ready');
  expect(inputs.source.table.id).not.toBe(inputs.target.table.id);
  expect(inputs.source.table.rowCount).toBe(4);
  expect(inputs.target.table.rowCount).toBe(4);
});

test('S1 numeric readback assertions fail on sparse or empty results', () => {
  expect(() => expectCatalogNumericRows(new Array<number>(2), [1, 0])).toThrow();
  expect(() => expectCatalogNumericRows([], [1, 0])).toThrow();
  expectCatalogNumericRows([1, 0], [1, 0]);
});

test('B7 scalar snapshots preserve signed zero and nonfinite missing-value sentinels', () => {
  const original = makeCatalogTableInput(
    {
      id: 'exact',
      label: 'Exact',
      rowCount: 4,
      columns: {value: [-0, NaN, Infinity, -Infinity]},
      columnTypes: {value: 'float64'}
    },
    {maximumRows: 4}
  );
  expect(
    parseCatalogTableInput(serializeCatalogTableInput(original), 4).table.columns['value']
  ).toEqual([-0, NaN, Infinity, -Infinity]);
});
