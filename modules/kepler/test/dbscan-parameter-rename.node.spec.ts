// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getCatalogBinding, normalizeCatalogParameters} from '../src/analysis/catalog-bindings';
import {resolveKeplerOperationParameters} from '../src/descriptors/default-parameters';
import {resolveKeplerOperationDescriptor} from '../src/descriptors/operation-descriptors';
import {getWorkflowOperationRegistry} from '../src/workflows/workflow-operations';
import {normalizeWorkflowParameters} from '../src/workflows/workflow-parameters';
import {planWorkflow} from '../src/workflows/workflow-plan';

const BINDING_IDS = ['density-clustering', 'curated-clusters-and-footprints'];
const RENAME_MESSAGE = 'minimumPoints was removed (use minimumSamples)';

test.each(BINDING_IDS)('%s rejects retired catalog parameters and default overrides', id => {
  const binding = getCatalogBinding(id)!;
  const {descriptor} = resolveKeplerOperationDescriptor(binding);
  for (const raw of [{minimumPoints: 7}, {minimumPoints: 7, minimumSamples: 8}] as Readonly<
    Record<string, number>
  >[]) {
    expect(() => normalizeCatalogParameters(binding, raw)).toThrow(`${id}: ${RENAME_MESSAGE}`);
    expect(() =>
      resolveKeplerOperationParameters(binding, descriptor, {}, {overrides: raw})
    ).toThrow(RENAME_MESSAGE);
  }
  expect(normalizeCatalogParameters(binding, {minimumSamples: 8})['minimumSamples']).toBe(8);
});

test.each(BINDING_IDS)('%s reports the rename in workflow validation and planning', id => {
  const registry = getWorkflowOperationRegistry();
  const operation = registry.get(id)!;
  expect(() =>
    normalizeWorkflowParameters(
      operation.parameters,
      {minimumPoints: 7},
      {operationTitle: operation.title}
    )
  ).toThrow(
    expect.objectContaining({
      code: 'unknown-parameter',
      details: {parameter: 'minimumPoints'},
      message: `${operation.title}: ${RENAME_MESSAGE}`
    })
  );
  const plan = planWorkflow(
    {
      schemaVersion: 1,
      id: 'clusters',
      title: 'Clusters',
      inputs: [],
      steps: [
        {
          id: 'cluster',
          operationId: id,
          inputs: {},
          parameters: {minimumPoints: {from: 'static', value: 7}},
          position: {x: 0, y: 0}
        }
      ],
      settings: {engine: 'cpu', autoRun: false}
    },
    {registry}
  );
  expect(plan.steps['cluster'].issues).toContainEqual({
    code: 'unknown-parameter',
    severity: 'error',
    stepId: 'cluster',
    parameter: 'minimumPoints',
    message: `${operation.title}: ${RENAME_MESSAGE}`
  });
});

test.each([
  'optics-ordering',
  'hierarchical-density-clustering'
])('%s already used minimumSamples', id => {
  expect(() => normalizeCatalogParameters(getCatalogBinding(id)!, {minimumPoints: 7})).toThrow(
    `${id}: minimumPoints is an unknown parameter; use a declared parameter.`
  );
});
