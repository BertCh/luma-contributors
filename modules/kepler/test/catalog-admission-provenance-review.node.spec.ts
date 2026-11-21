// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  normalizeCatalogParameters,
  validateCatalogInputs
} from '../src/analysis/catalog-bindings';
import {resolveKeplerOperationParameters} from '../src/descriptors/default-parameters';
import {resolveKeplerOperationDescriptor} from '../src/descriptors/operation-descriptors';
import {evaluateKeplerOperationApplicability} from '../src/descriptors/applicability';
import {makeRasterEmbeddingCatalogOutput} from '../src/analysis/catalog-raster-embeddings-bindings';
import {resolveCatalogParameterUnit} from '../src/analysis/catalog-parameter-constraints';
import {
  makeWorkflowOperationFromCatalogBinding,
  getWorkflowOperationRegistry
} from '../src/workflows/workflow-operations';
import {
  getWorkflowParameterProblem,
  normalizeWorkflowParameters
} from '../src/workflows/workflow-parameters';
import {planWorkflow} from '../src/workflows/workflow-plan';
import {getWorkbenchParameterRows} from '../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-model';
import {WorkbenchParameterForm} from '../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-form';
import {createWorkbenchStore} from '../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {makeFixtureHost} from './workflows/workflow-fixtures';
import {buildTableProfile} from './descriptors/engine-test-profiles';
import type {WorkflowDocument, WorkflowStep} from '../src/workflows/workflow-types';
import type {
  CatalogParameterSpec,
  CatalogParameterValue
} from '../src/analysis/catalog-binding-types';

const hostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

test('B2 ICP control resolves a maximumIterations default of 10', () => {
  const operation = makeWorkflowOperationFromCatalogBinding(getCatalogBinding('pointcloud-icp')!);
  const step: WorkflowStep = {
    id: 'icp',
    operationId: operation.id,
    inputs: {},
    parameters: {},
    position: {x: 0, y: 0}
  };
  const document: WorkflowDocument = {
    schemaVersion: 1,
    id: 'controls',
    title: 'Controls',
    inputs: [],
    steps: [step],
    settings: {engine: 'auto', autoRun: false}
  };
  const rows = getWorkbenchParameterRows(document, step, operation, {
    profilesByInput: {},
    executionEngine: 'gpu'
  });
  expect(rows.find(row => row.spec.name === 'maximumIterations')?.value).toBe(10);
  expect(rows.some(row => row.spec.name === 'iterations')).toBe(false);
});

test('B2 a retired ICP parameter blocks the workflow plan', () => {
  const step: WorkflowStep = {
    id: 'icp',
    operationId: 'pointcloud-icp',
    inputs: {},
    parameters: {
      maximumIterations: {from: 'static', value: 8},
      iterations: {from: 'static', value: 6}
    },
    position: {x: 0, y: 0}
  };
  const document: WorkflowDocument = {
    schemaVersion: 1,
    id: 'conflict',
    title: 'Conflict',
    inputs: [],
    steps: [step],
    settings: {engine: 'auto', autoRun: false}
  };
  const plan = planWorkflow(document, {registry: getWorkflowOperationRegistry()});
  expect(plan.steps['icp'].issues).toContainEqual(
    expect.objectContaining({
      code: 'unknown-parameter',
      parameter: 'iterations',
      message: expect.stringMatching(/iterations was removed; use maximumIterations/)
    })
  );
});

test('B3 fully mapped column tables are ready without vertex metadata', () => {
  const profile = buildTableProfile(
    'forecast',
    ['observed', 'predicted', 'variance', 'member1', 'member2', 'pvalue'].map(name => ({
      name,
      type: 'number' as const,
      rowCount: 3,
      nullCount: 0,
      minimum: 0.01,
      maximum: 0.2,
      distinctCount: 3
    })),
    3
  );
  for (const id of ['validation-gaussian-scores', 'validation-ensemble-scores', 'validation-fdr']) {
    const binding = getCatalogBinding(id)!;
    const overrides: Record<string, CatalogParameterValue> =
      id === 'validation-gaussian-scores'
        ? {observationProperty: 'observed', meanProperty: 'predicted', varianceProperty: 'variance'}
        : id === 'validation-ensemble-scores'
          ? {observationProperty: 'observed', memberProperties: 'member1,member2'}
          : {pValueProperty: 'pvalue'};
    const result = evaluateKeplerOperationApplicability(
      binding,
      resolveKeplerOperationDescriptor(binding).descriptor,
      [profile],
      {executionEngine: 'gpu', overrides}
    );
    expect(result.missing).toEqual([]);
    expect(result.status).toBe('ready');
  }
});

test('B4 score and clip ports declare geometry inherited from their selected input', () => {
  for (const id of ['validation-gaussian-scores', 'validation-ensemble-scores', 'validation-fdr'])
    expect(getCatalogBinding(id)!.outputs[0]).toHaveProperty('geometryFromInput', 'points');
  expect(getCatalogBinding('geometry-convex-clip')!.outputs[0]).toHaveProperty(
    'geometryFromInput',
    'features'
  );
});

test('B5 dual-route bindings retain callable feature forwarding methods', () => {
  for (const id of ['validation-gaussian-scores', 'validation-ensemble-scores', 'validation-fdr']) {
    const binding = getCatalogBinding(id)!;
    expect(binding.runGpu).toBeTypeOf('function');
    expect(() =>
      validateCatalogInputs(binding, {
        points: {
          type: 'FeatureCollection',
          features: [{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {}}]
        }
      })
    ).not.toThrow();
  }
});

test('B6 rendered units follow profile-derived parameter rows without a run override', async () => {
  const source: WorkflowStep = {
    id: 'source',
    operationId: 'source-geojson',
    inputs: {},
    parameters: {
      coordinateSystem: {from: 'static', value: 'planar'},
      text: {
        from: 'static',
        value: JSON.stringify({
          type: 'FeatureCollection',
          features: [0, 1, 2].map(index => ({
            type: 'Feature',
            geometry: {type: 'Point', coordinates: [index * 100, 0]},
            properties: {v: index + 1}
          }))
        })
      }
    },
    position: {x: 0, y: 0}
  };
  const step: WorkflowStep = {
    id: 'idw',
    operationId: 'idw-grid',
    inputs: {points: {stepId: 'source', output: 'data'}},
    parameters: {valueProperty: {from: 'static', value: 'v'}},
    position: {x: 1, y: 0}
  };
  const document: WorkflowDocument = {
    schemaVersion: 1,
    id: 'units',
    title: 'Units',
    inputs: [],
    steps: [source, step],
    settings: {engine: 'cpu', autoRun: false}
  };
  const store = createWorkbenchStore({
    host: makeFixtureHost(),
    registry: getWorkflowOperationRegistry()
  });
  try {
    store.addWorkflow(document);
    const markup = renderToStaticMarkup(
      createElement(WorkbenchParameterForm, {store, stepId: 'idw'})
    );
    expect(markup).toContain('coordinate units</span>');
    expect(markup).not.toContain('metres</span>');
  } finally {
    await store.clear();
  }
});

test('B7 unaudited CRS units retain their legacy label', () => {
  expect(resolveCatalogParameterUnit({unit: 'crs'}, 'EPSG:4326')).toBe('CRS units');
  expect(resolveCatalogParameterUnit({unit: 'source-coordinate-units'}, 'EPSG:4326')).toBe(
    'source coordinate units'
  );
});

test('B8 descriptor values capture registry defaults and explicit override lists', () => {
  const binding = getCatalogBinding('network-service-area')!;
  const descriptor = resolveKeplerOperationDescriptor(binding).descriptor;
  const before = [...(getCatalogDefaultParameters(binding)['budgets'] as readonly number[])];
  const defaults = resolveKeplerOperationParameters(binding, descriptor, {});
  try {
    (defaults.parameters['budgets'] as number[])[0] = 21;
    expect(getCatalogDefaultParameters(binding)['budgets']).toEqual(before);
  } finally {
    // Leave the shared registry untouched even when the pre-fix regression fails.
    (binding.parameters.find(spec => spec.name === 'budgets')!.defaultValue as number[])[0] =
      before[0];
  }
  const override = [50, 100];
  const resolved = resolveKeplerOperationParameters(
    binding,
    descriptor,
    {},
    {overrides: {budgets: override}}
  );
  override[0] = 999;
  expect(resolved.parameters['budgets']).toEqual([50, 100]);
});

test('B9 explicit stage evidence preserves CPU work and reconciles only a codec prefix', () => {
  const stages = [
    {
      identity: {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'getRasterEmbeddingStretchRanges'
      },
      execution: 'cpu' as const
    },
    {
      identity: {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPURasterEmbeddingProjection'
      },
      execution: 'gpu' as const
    }
  ];
  const output = makeRasterEmbeddingCatalogOutput({
    width: 1,
    height: 1,
    rowCount: 1,
    layout: 'pixels',
    columns: {},
    contributors: ['GPURasterEmbeddingQuantize', ...stages.map(stage => stage.identity.exportName)],
    stages
  });
  expect(output.stages?.slice(1)).toEqual(stages);
  expect(output.dispatchedContributors).toEqual([
    'GPURasterEmbeddingQuantize',
    'GPURasterEmbeddingProjection'
  ]);
});

test('B10 scalar and list numeric grammar agrees and retired ICP parameters are rejected', () => {
  expect(() =>
    normalizeCatalogParameters(getCatalogBinding('idw-grid')!, {cellSize: '0x10'})
  ).toThrow();
  const list: {id: string; parameters: CatalogParameterSpec[]} = {
    id: 'lists',
    parameters: [{name: 'values', label: 'Values', kind: 'number-list'}]
  };
  expect(() => normalizeCatalogParameters(list, {values: '0x10,2'})).toThrow();
  const binding = getCatalogBinding('pointcloud-icp')!;
  for (const iterations of [true, [1], null, '0x1']) {
    expect(() => normalizeCatalogParameters(binding, {maximumIterations: 1, iterations})).toThrow(
      /iterations was removed; use maximumIterations/
    );
    expect(() =>
      normalizeWorkflowParameters(
        binding.parameters,
        {maximumIterations: 1, iterations},
        {operationTitle: binding.title}
      )
    ).toThrow(/iterations was removed; use maximumIterations/);
  }
});

test('S2 validation identity is unpunctuated and integer failures have integer codes', () => {
  try {
    normalizeCatalogParameters(getCatalogBinding('idw-grid')!, {cellSize: -1});
    throw new Error('expected refusal');
  } catch (error) {
    expect(error).toMatchObject({id: 'idw-grid', code: 'NUMBER_RANGE', received: -1});
  }
  try {
    normalizeCatalogParameters(getCatalogBinding('pointcloud-icp')!, {maximumIterations: 1.5});
    throw new Error('expected refusal');
  } catch (error) {
    expect(error).toMatchObject({id: 'pointcloud-icp', code: 'INTEGER_RANGE'});
  }
});

test('S3 FTLE admission excludes a duration that underflows Float32', () => {
  const binding = getCatalogBinding('raster-lagrangian-ftle')!;
  expect(() => normalizeCatalogParameters(binding, {timeStep: 1e-50})).toThrow('Float32');
  expect(getWorkflowParameterProblem(binding.parameters[0], 1e-50)).toContain('Float32');
  // A nonzero scaled duration cannot restore a time step rounded to signed zero.
  for (const timeStep of [-1e-46, 1e-46]) {
    expect(() => normalizeCatalogParameters(binding, {timeStep})).toThrow('Float32');
    expect(getWorkflowParameterProblem(binding.parameters[0], timeStep)).toContain('Float32');
  }
  for (const timeStep of [-0.1, 0.1])
    expect(normalizeCatalogParameters(binding, {timeStep})['timeStep']).toBe(timeStep);
});

test('S4 workflow errors use the same resolved physical unit as their controls', () => {
  const binding = getCatalogBinding('idw-grid')!;
  expect(() =>
    normalizeWorkflowParameters(
      binding.parameters,
      {coordinateSystem: 'planar', cellSize: -1},
      {operationTitle: binding.title}
    )
  ).toThrow('coordinate units');
  expect(() =>
    normalizeWorkflowParameters(
      binding.parameters,
      {coordinateSystem: 'EPSG:4326', cellSize: -1},
      {operationTitle: binding.title}
    )
  ).toThrow('metres');
});
