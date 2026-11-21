// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {expectCatalogNumericRows} from './workflow-catalog-numeric-assertions';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {getCatalogBinding} from '../../src/analysis/catalog-bindings';
import {makeWorkflowOperationFromCatalogBinding} from '../../src/workflows/workflow-operations';
import {makeWorkflowData, makeWorkflowDataFromFeatures} from '../../src/workflows/workflow-data';
import type {WorkflowRunContext} from '../../src/workflows/workflow-types';
import {createCatalogIcpWorkflowInputs} from './workflow-catalog-icp-fixture';
import {makeFixtureHost} from './workflow-fixtures';
import {readCatalogNativeInput} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-native-input';
import {
  restoreAnalysisCatalogInvocation,
  editAnalysisCatalogOverride
} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-form-model';

test('W5-01 restored and edited ICP budget executes eight GPU history rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding('pointcloud-icp')!;
  const operation = makeWorkflowOperationFromCatalogBinding(binding);
  const restored = restoreAnalysisCatalogInvocation(
    JSON.stringify({
      schemaVersion: 1,
      operationId: binding.id,
      inputs: {},
      parameters: {
        maximumIterations: 13,
        coordinateSystem: 'planar',
        maximumDistance: 2
      }
    })
  ).invocation.parameters;
  const edited = editAnalysisCatalogOverride(binding, restored, 'maximumIterations', 8);
  const inputs = createCatalogIcpWorkflowInputs();
  const result = await operation.run(
    {
      device,
      host: makeFixtureHost(),
      signal: new AbortController().signal,
      engine: 'auto',
      workflowId: 'icp',
      stepId: 'icp'
    },
    inputs,
    operation.normalizeParameters(edited)
  );
  const history = result.outputs['registration'].table;
  expect(result.engine).toBe('gpu');
  expect(history.rowCount).toBe(8);
  // Identity tetrahedra give four exact matches in every executed or frozen history row.
  expect(history.columns['matched_count']).toEqual([4, 4, 4, 4, 4, 4, 4, 4]);
  expect(history.columns['registration_status']).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
});

test('UCO-13 native two-row score and FDR tables execute and point alternatives retain both geometries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fields = ['observed', 'predicted', 'variance', 'member1', 'member2', 'pvalue'];
  const rows = [
    [1, 2, 0, 2, 2, 0.01],
    [3, 3, 0, 3, 3, 0.2]
  ];
  const native = await readCatalogNativeInput({
    source: {
      fields: fields.map(name => ({name})),
      dataContainer: {numRows: () => 2, valueAt: (row, column) => rows[row][column]}
    },
    field: '',
    acceptsTable: true,
    isTrajectory: false,
    parameters: {}
  });
  if (native.type !== 'table') throw new Error('Expected a native table');
  const table = makeWorkflowData(native.table);
  const points = makeWorkflowDataFromFeatures(
    {
      type: 'FeatureCollection',
      features: rows.map((row, index) => ({
        geometry: {type: 'Point', coordinates: [index * 10, index]},
        properties: Object.fromEntries(fields.map((name, column) => [name, row[column]]))
      }))
    },
    {id: 'forecasts', label: 'Forecasts'}
  );
  const context: WorkflowRunContext = {
    device,
    host: makeFixtureHost(),
    signal: new AbortController().signal,
    engine: 'auto',
    workflowId: 'scores',
    stepId: 'scores'
  };
  for (const id of ['validation-gaussian-scores', 'validation-ensemble-scores', 'validation-fdr']) {
    const binding = getCatalogBinding(id)!;
    const operation = makeWorkflowOperationFromCatalogBinding(binding);
    const parameters = operation.normalizeParameters(
      id === 'validation-ensemble-scores'
        ? {memberProperties: 'member1,member2'}
        : id === 'validation-fdr'
          ? {method: 'bonferroni', alpha: 0.05}
          : {}
    );
    const tableResult = await operation.run(context, {points: table}, parameters);
    const featureResult = await operation.run(context, {points}, parameters);
    const output = tableResult.outputs[binding.outputs[0].tableId];
    const featureOutput = featureResult.outputs[binding.outputs[0].tableId];
    expect(tableResult.engine).toBe('gpu');
    expect(output.kind).toBe('table');
    expect(featureOutput.kind).toBe('point');
    expect(output.table.rowCount).toBe(2);
    expect(featureOutput.table.rowCount).toBe(2);
    expect(output.table.geometryColumn).toBeUndefined();
    expect(
      featureOutput.table.columns[featureOutput.table.geometryColumn!].map(value =>
        JSON.parse(String(value))
      )
    ).toEqual([
      {type: 'Point', coordinates: [0, 0]},
      {type: 'Point', coordinates: [10, 1]}
    ]);
    // Point forecasts give CRPS |prediction-observation|; Bonferroni multiplies by two tests.
    const expected = id === 'validation-fdr' ? [0.02, 0.4] : [1, 0];
    const column = id === 'validation-fdr' ? 'adjusted_pvalue' : 'crps';
    for (const result of [output, featureOutput]) {
      const values = result.table.columns[column];
      expectCatalogNumericRows(values, expected);
    }
  }
});
