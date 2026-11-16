// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {test} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {createWorkflowDocument} from '../../src/workflows/workflow-document';
import {addWorkflowStep} from '../../src/workflows/workflow-edit';
import {getWorkflowOperationRegistry} from '../../src/workflows/workflow-operations';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {encodeKeplerResultPresentation} from '../../src/presentation/result-presentation';
import type {WorkflowData} from '../../src/workflows/workflow-types';
import {makeFixtureHost} from './workflow-fixtures';

// The browser runner must throw the evidence: console output is not collected by headless CI.
test('AX GPU workbench output retains the report and exact classification used for publication', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const registry = getWorkflowOperationRegistry();
  const host = makeFixtureHost();
  const published: WorkflowData[] = [];
  const publishOutput = host.publishOutput!;
  host.publishOutput = async (request, signal) => {
    published.push(request.data);
    return publishOutput(request, signal);
  };
  host.getDevice = () => device;
  const source = addWorkflowStep(createWorkflowDocument({title: 'GPU AX integration'}), registry, {
    operationId: 'source-geojson',
    parameters: {
      coordinateSystem: 'EPSG:4326',
      text: JSON.stringify({
        type: 'FeatureCollection',
        features: Array.from({length: 12}, (_, index) => ({
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'Point',
            coordinates: [-87.6 + (index % 4) * 0.001, 41.8 + Math.floor(index / 4) * 0.001]
          }
        }))
      })
    }
  });
  const density = addWorkflowStep(source.document, registry, {
    operationId: 'kernel-density-grid',
    parameters: {cellSize: 100, bandwidth: 200},
    connectFrom: {stepId: source.stepId, output: 'data'},
    connectTo: 'points'
  });
  const document = density.document;
  document.steps.find(step => step.id === density.stepId)!.display = ['density-cells'];
  const runner = new WorkflowRunner({registry, host});
  try {
    runner.setDocument(document);
    const record = await runner.run();
    const state = runner.getState().steps[density.stepId];
    const output = state.result?.outputs['density-cells'];
    if (
      state.status !== 'done' ||
      state.result?.engine !== 'gpu' ||
      !state.result.dispatchedContributors.length ||
      !output?.analysisReport ||
      !output.analysisPresentation?.presentation
    ) {
      throw new Error(`AX GPU integration failed: ${JSON.stringify({record, state})}`);
    }
    const encoding = encodeKeplerResultPresentation(output.table, output.analysisPresentation)!;
    const counts = encoding.legend.map(
      (_, index) => encoding.values.filter(value => value === encoding.categories[index]).length
    );
    const expected = encoding.legend.map(entry => entry.count);
    if (
      JSON.stringify(counts) !== JSON.stringify(expected) ||
      host.published.length !== 1 ||
      published[0].analysisPresentation?.presentation?.breaks.join(',') !==
        output.analysisPresentation.presentation.breaks.join(',')
    ) {
      throw new Error(
        `AX GPU publication differs from legend: ${JSON.stringify({counts, expected, publications: host.published})}`
      );
    }
  } finally {
    await runner.destroy();
  }
});
