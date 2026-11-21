// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {AnalysisSheetParameter} from '../../../../examples/deck/kepler-compute/src/analysis-sheet';
import {createAnalysisSheetModel} from '../../../../examples/deck/kepler-compute/src/analysis-sheet-model';
import {expect, test} from 'vitest';
import {formatWorkbenchParameterHint} from '../../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-model';
import {
  getCatalogNumberProblem,
  resolveCatalogParameterUnit
} from '../../src/analysis/catalog-parameter-constraints';
import {readCatalogNativeInput} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-native-input';
import {
  createAnalysisCatalogFormModel,
  profileAnalysisCatalogInput,
  canonicalizeAnalysisCatalogOverrides,
  editAnalysisCatalogOverride
} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-form-model';
import {
  getCatalogBinding,
  parseCatalogInvocation,
  serializeCatalogInvocation,
  normalizeCatalogParameters
} from '../../src/analysis/catalog-bindings';
import {
  makeWorkflowOperationFromCatalogBinding,
  getWorkflowOperationRegistry,
  getWorkflowPortCompatibility,
  createWorkflowOperationRegistry
} from '../../src/workflows/workflow-operations';
import {resolveWorkflowParameters, planWorkflow} from '../../src/workflows/workflow-plan';
import type {WorkflowDocument, WorkflowStep} from '../../src/workflows/workflow-types';

import {makeWorkflowFromCatalogHistory} from '../../src/workflows/workflow-export';
import {estimateWorkflowPreflight} from '../../src/workflows/workflow-preflight';
import {makeWorkflowDataFromFeatures, getWorkflowFeatures} from '../../src/workflows/workflow-data';
import {makeFixtureHost, makeFixtureRegistry} from './workflow-fixtures';
import {resolveKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';

const hostRequire = createRequire(
  new URL('../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

function makeDocument(parameters: WorkflowStep['parameters']): WorkflowDocument {
  return {
    schemaVersion: 1,
    id: 'icp',
    title: 'ICP',
    inputs: [{id: 'budget', label: 'Budget', kind: 'integer', value: 7}],
    settings: {engine: 'auto', autoRun: false},
    steps: [
      {id: 'icp', operationId: 'pointcloud-icp', inputs: {}, parameters, position: {x: 0, y: 0}}
    ]
  };
}

test('W3-01 resolves explicit ICP budgets and defaults', () => {
  const operation = makeWorkflowOperationFromCatalogBinding(getCatalogBinding('pointcloud-icp')!);
  const parameterCases: WorkflowStep['parameters'][] = [
    {maximumIterations: {from: 'static', value: 7}},
    {maximumIterations: {from: 'input', inputId: 'budget'}},
    {}
  ];
  for (const parameters of parameterCases) {
    const document = makeDocument(parameters);
    const values = resolveWorkflowParameters(document, document.steps[0], operation);
    const expected = Object.keys(parameters).length ? 7 : 10;
    expect(values['maximumIterations']).toBe(expected);
    expect(values).not.toHaveProperty('iterations');
    expect(operation.normalizeParameters(values)['maximumIterations']).toBe(expected);
  }
  expect(() => operation.normalizeParameters({iterations: 6})).toThrow(
    /iterations was removed; use maximumIterations/
  );
});

test('W5-01 retired parameter keys become form problems instead of throwing during render', () => {
  const binding = getCatalogBinding('pointcloud-icp')!;
  const model = createAnalysisCatalogFormModel(
    binding,
    {},
    {overrides: {maximumIterations: 8, iterations: 6}}
  );
  expect(model.status).toBe('blocked');
  expect(model.problems.join(' ')).toMatch(/iterations was removed; use maximumIterations/);
});

test('UCO-13 column-score adapters admit table and feature alternatives', async () => {
  const binding = getCatalogBinding('validation-gaussian-scores')!;
  const operation = makeWorkflowOperationFromCatalogBinding(binding);
  expect(operation.inputs[0].accepts).toEqual(['point', 'table']);
  expect(operation.outputs[0]).toHaveProperty('kindFrom', 'points');
  const source = {
    fields: [{name: 'observed'}, {name: 'predicted'}, {name: 'variance'}],
    dataContainer: {
      numRows: () => 2,
      valueAt: (row: number, column: number) =>
        [
          [1, 1, 0],
          [3, 1, 0]
        ][row][column]
    }
  };
  const table = await readCatalogNativeInput({
    source,
    field: '',
    isTrajectory: false,
    acceptsTable: true,
    parameters: {}
  });
  expect(table.type).toBe('table');
  if (table.type !== 'table') throw new Error('Expected a native table');
  expect(table.table.rowCount).toBe(2);
  expect(table.table.columns['observed']).toEqual([1, 3]);
  const profile = profileAnalysisCatalogInput(binding, 'points', table);
  expect(profile.rowCount).toBe(2);
  expect(profile.kind).toBe('attributes');
  const promoted = makeWorkflowFromCatalogHistory(
    [
      {
        id: 'scores',
        operationId: binding.id,
        invocation: serializeCatalogInvocation(binding, {}, {points: table}),
        inputs: {points: {mode: 'native', datasetId: 'forecasts', field: ''}},
        engine: 'auto'
      }
    ],
    {registry: getWorkflowOperationRegistry()}
  );
  expect(
    promoted.steps.find(step => step.operationId === 'source-dataset')?.parameters['geometryMode']
  ).toEqual({from: 'static', value: 'none'});
  const featureInput = await readCatalogNativeInput({
    source: {
      fields: [{name: 'geometry'}, {name: 'observed'}],
      dataContainer: {
        numRows: () => 2,
        valueAt: (row, column) =>
          column === 0 ? {type: 'Point', coordinates: [row * 10, row]} : row + 1
      }
    },
    field: 'geometry',
    acceptsTable: true,
    isTrajectory: false,
    parameters: {}
  });
  if (featureInput.type !== 'FeatureCollection') throw new Error('Expected feature transport');
  expect(featureInput.features.map(feature => feature.geometry)).toEqual([
    {type: 'Point', coordinates: [0, 0]},
    {type: 'Point', coordinates: [10, 1]}
  ]);
});

test('UCO-15 numeric exclusions remain named in catalog problems and workbench hints', () => {
  const binding = getCatalogBinding('raster-lagrangian-ftle')!;
  const operation = makeWorkflowOperationFromCatalogBinding(binding);
  const specification = operation.parameters.find(parameter => parameter.name === 'timeStep')!;
  const hint = formatWorkbenchParameterHint(specification);
  expect(hint).toContain('unscaled value must remain finite and nonzero as Float32');
  expect(hint).toContain('value multiplied by 10 must remain finite and nonzero as Float32');
  const underflowingStep = -(2 ** -151);
  expect(Math.fround(underflowingStep)).toBe(-0);
  expect(Math.fround(10 * underflowingStep)).toBe(-(2 ** -148));
  for (const value of [0, 1e-50, 1e-46, underflowingStep]) {
    const form = createAnalysisCatalogFormModel(binding, {}, {overrides: {timeStep: value}});
    expect(form.runDisabled).toBe(true);
    const requirement =
      value === 0 ? 'be nonzero' : 'remain finite and nonzero as Float32 before duration scaling';
    expect(form.problems.join(' ')).toContain(
      `raster-lagrangian-ftle: timeStep must ${requirement}; received ${value}`
    );
  }
  // The FTLE range prevents duration overflow, so exercise the scale constraint on its own.
  const largeStep = 2 ** 126;
  expect(Math.fround(largeStep)).toBe(largeStep);
  expect(
    getCatalogNumberProblem({nonzeroFloat32Scale: specification.nonzeroFloat32Scale}, largeStep)
  ).toBe(`must give a finite nonzero Float32 value when multiplied by 10; received ${largeStep}`);
  for (const timeStep of [-0.1, 0.1]) {
    const form = createAnalysisCatalogFormModel(binding, {}, {overrides: {timeStep}});
    expect(form.problems).toEqual([]);
    expect(form.parameters['timeStep']).toBe(timeStep);
  }
  expect(resolveCatalogParameterUnit({unit: 'crs'}, 'planar')).toBe('CRS units');
});

test('W5-01 restore then edit/reset keeps one rendered ICP control and an executable budget', () => {
  const binding = getCatalogBinding('pointcloud-icp')!;
  const saved = parseCatalogInvocation(
    serializeCatalogInvocation(
      binding,
      {maximumIterations: 6},
      {
        source: cloud,
        target: cloud
      }
    )
  );
  const restored = canonicalizeAnalysisCatalogOverrides(binding, saved.invocation.parameters);
  expect(restored['maximumIterations']).toBe(6);
  expect(restored).not.toHaveProperty('iterations');
  const edited = editAnalysisCatalogOverride(binding, restored, 'maximumIterations', 8);
  const form = createAnalysisCatalogFormModel(binding, {}, {overrides: edited});
  expect(form.problems).toEqual([]);
  const sheet = createAnalysisSheetModel(binding, form, {});
  const markup = renderToStaticMarkup(
    createElement(
      'div',
      {},
      ...sheet.rows.map(row =>
        createElement(AnalysisSheetParameter, {
          key: row.specification.name,
          spec: binding.parameters.find(parameter => parameter.name === row.specification.name)!,
          row,
          value: row.value,
          source: row.source,
          onChange: () => {}
        })
      )
    )
  );
  expect(markup).toContain('catalog-parameter-maximumIterations');
  expect(markup).not.toContain('catalog-parameter-iterations');
  expect(normalizeCatalogParameters(binding, form.parameters)['maximumIterations']).toBe(8);
  const reset = editAnalysisCatalogOverride(
    binding,
    {maximumIterations: 8},
    'maximumIterations',
    undefined
  );
  expect(normalizeCatalogParameters(binding, reset)['maximumIterations']).toBe(10);
});

test('W3-01 ICP descriptor and preflight receive maximumIterations budgets', () => {
  const binding = getCatalogBinding('pointcloud-icp')!;
  const descriptor = resolveKeplerOperationDescriptor(binding).descriptor;
  expect(validateKeplerOperationDescriptor(descriptor, binding)).toEqual([]);
  expect(
    descriptor.parameters.find(parameter => parameter.name === 'maximumIterations')?.level
  ).toBe('basic');
  const document = makeDocument({maximumIterations: {from: 'static', value: 7}});
  const estimate = estimateWorkflowPreflight(document, getWorkflowOperationRegistry());
  expect(estimate.steps[0].reason).not.toContain('invalid parameters');
  // ICP has no GPU cost model yet; accepting its budget does not fabricate a zero-cost estimate.
  expect(estimate.steps[0].status).toBe('unknown');
});

test('UCO-13 clipped line outputs retain their actual geometry through execution and planning', async () => {
  const binding = getCatalogBinding('geometry-convex-clip')!;
  const operation = makeWorkflowOperationFromCatalogBinding(binding);
  expect(operation.outputs[0]).toHaveProperty('kindFrom', 'features');
  const features = makeWorkflowDataFromFeatures(
    {
      type: 'FeatureCollection',
      features: [
        {
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
    },
    {id: 'line', label: 'Line'}
  );
  const clip = makeWorkflowDataFromFeatures(
    {
      type: 'FeatureCollection',
      features: [
        {
          geometry: {
            type: 'Polygon',
            coordinates: [
              [
                [0, 0],
                [2, 0],
                [2, 2],
                [0, 2],
                [0, 0]
              ]
            ]
          },
          properties: {}
        }
      ]
    },
    {id: 'clip', label: 'Clip'}
  );
  features.table.coordinateSystem = 'planar';
  clip.table.coordinateSystem = 'planar';
  const result = await operation.run(
    {
      device: null,
      engine: 'cpu',
      host: makeFixtureHost(),
      signal: new AbortController().signal,
      workflowId: 'clip',
      stepId: 'clip'
    },
    {features, clip},
    operation.normalizeParameters({})
  );
  const output = result.outputs[binding.outputs[0].tableId];
  expect(output.kind).toBe('line');
  expect(getWorkflowFeatures(output).features[0].geometry).toEqual({
    type: 'LineString',
    coordinates: [
      [0, 1],
      [2, 1]
    ]
  });
  expect(
    getWorkflowPortCompatibility(
      'line',
      makeWorkflowOperationFromCatalogBinding(getCatalogBinding('validation-fdr')!).inputs[0]
    )
  ).toBe('no');
});

const cloud = {
  type: 'FeatureCollection' as const,
  features: [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ].map(coordinates => ({geometry: {type: 'Point' as const, coordinates}, properties: {}}))
};

test('UCO-13 planning follows inherited table and line kinds into the next consumer', () => {
  const fixtures = makeFixtureRegistry();
  const score = makeWorkflowOperationFromCatalogBinding(getCatalogBinding('validation-fdr')!);
  const clip = makeWorkflowOperationFromCatalogBinding(getCatalogBinding('geometry-convex-clip')!);
  const consumer = fixtures.get('fixture-count')!;
  const tableConsumer = {
    ...consumer,
    id: 'table-consumer',
    inputs: [
      {
        name: 'rows',
        label: 'Rows',
        accepts: ['table' as const],
        geometryFree: true,
        maximumRows: 2048
      }
    ]
  };
  const lineConsumer = {
    ...consumer,
    id: 'line-consumer',
    inputs: [{name: 'rows', label: 'Rows', accepts: ['line' as const], maximumRows: 2048}]
  };
  const registry = createWorkflowOperationRegistry([
    ...fixtures.list(),
    score,
    clip,
    tableConsumer,
    lineConsumer
  ]);
  for (const fixture of [
    {producer: score, input: 'points', kind: 'table' as const, consumer: tableConsumer},
    {producer: clip, input: 'features', kind: 'line' as const, consumer: lineConsumer}
  ]) {
    const document = makeDocument({});
    document.steps = [
      {
        id: 'source',
        operationId: 'fixture-source',
        inputs: {},
        parameters: {},
        position: {x: 0, y: 0}
      },
      {
        id: 'producer',
        operationId: fixture.producer.id,
        inputs: {[fixture.input]: {stepId: 'source', output: 'data'}},
        parameters: {},
        position: {x: 1, y: 0}
      },
      {
        id: 'consumer',
        operationId: fixture.consumer.id,
        inputs: {rows: {stepId: 'producer', output: fixture.producer.outputs[0].name}},
        parameters: {},
        position: {x: 2, y: 0}
      }
    ];
    const plan = planWorkflow(document, {registry, observedKinds: {source: {data: fixture.kind}}});
    expect(
      plan.steps['consumer'].issues.filter(issue =>
        ['incompatible-port', 'unverified-port'].includes(issue.code)
      )
    ).toEqual([]);
  }
});
