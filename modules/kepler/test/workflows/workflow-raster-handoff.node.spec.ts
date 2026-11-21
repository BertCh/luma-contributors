// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {expect, test, vi} from 'vitest';
import {
  CURATED_SURFACE_PREPARATION_BINDING,
  prepareSurfacePreparation,
  publishSurfacePreparation
} from '../../src/curated/bindings/curated-surface-preparation-binding';
import {
  CURATED_DRAINAGE_NETWORK_BINDING,
  prepareCuratedDrainageNetwork
} from '../../src/curated/bindings/curated-drainage-network-binding';
import {createDrainageNetworkSample} from '../../src/curated/samples/drainage-network';
import {
  getWorkflowCatalogInput,
  getWorkflowDataKind,
  makeWorkflowData
} from '../../src/workflows/workflow-data';
import {
  getWorkflowPortCompatibility,
  makeWorkflowOperationFromCatalogBinding
} from '../../src/workflows/workflow-operations';
import {planWorkflow} from '../../src/workflows/workflow-plan';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {createRasterHandoffFixture} from './workflow-raster-handoff-fixtures';

function publishFixture(surface: ReturnType<typeof createDrainageNetworkSample>['elevation']) {
  const {settings} = prepareSurfacePreparation({surface}, {maximumDistance: 1, method: 'nearest'});
  return publishSurfacePreparation(surface, settings, {
    values: new Float32Array([9, 8, 7, 6, 8, 4, 3, 2, 1]),
    validity: new Uint32Array(9).fill(1),
    originalValidity: Uint32Array.from(surface.band.validity),
    filled: new Uint32Array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
    filtered: new Uint32Array(9),
    unresolvedCount: 0
  });
}

test('automatic wrapping keeps retained raster samples ahead of planar or polygon display tables', () => {
  const {elevation} = createDrainageNetworkSample();
  const table = publishFixture(elevation).tables[0];
  for (const display of [
    table,
    {
      ...table,
      geometryColumn: 'geometry',
      columns: {
        ...table.columns,
        geometry: Array(9).fill('{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}')
      }
    }
  ]) {
    const data = makeWorkflowData(display);
    expect(getWorkflowDataKind(display)).toBe('raster');
    expect(data.kind).toBe('raster');
    expect(data.table).toBe(display);
    expect(data.raster).toBe(display.raster);
    expect(getWorkflowCatalogInput(data, 0)).toBe(display.raster);
    expect(data.raster?.grid).toBe(elevation.grid);
    expect(data.raster?.crs).toBe('planar');
  }
  const preparation = makeWorkflowOperationFromCatalogBinding(CURATED_SURFACE_PREPARATION_BINDING);
  const drainage = makeWorkflowOperationFromCatalogBinding(CURATED_DRAINAGE_NETWORK_BINDING);
  expect(preparation.outputs[0].kind).toBe('raster');
  expect(getWorkflowPortCompatibility(preparation.outputs[0].kind, drainage.inputs[0])).toBe('yes');
});

// GPU dispatches are stubbed here; the companion headless spec qualifies actual computation.
test('the workflow runner automatically hands prepared planar raster samples to drainage', async () => {
  const prepareDispatch = vi.fn<NonNullable<typeof CURATED_SURFACE_PREPARATION_BINDING.runDataGpu>>(
    async (_context, inputs) => {
      const prepared = prepareSurfacePreparation(inputs, {maximumDistance: 1, method: 'nearest'});
      return publishFixture(prepared.surface);
    }
  );
  const drainageDispatch = vi.fn<NonNullable<typeof CURATED_DRAINAGE_NETWORK_BINDING.runDataGpu>>(
    async (_context, inputs, values) => {
      const prepared = prepareCuratedDrainageNetwork(inputs, values);
      expect(Array.from(prepared.raster.band.values)).toEqual([9, 8, 7, 6, 8, 4, 3, 2, 1]);
      expect(Array.from(prepared.raster.band.validity)).toEqual(Array(9).fill(1));
      expect(prepared.raster.crs).toBe('planar');
      return {
        tables: CURATED_DRAINAGE_NETWORK_BINDING.outputs.map(output => ({
          id: output.tableId,
          label: output.label,
          rowCount: 0,
          columns: {},
          columnTypes: {}
        })),
        diagnostics: [],
        summary: {}
      };
    }
  );
  const {host, registry, document, elevation} = createRasterHandoffFixture(
    {} as Device,
    {...CURATED_SURFACE_PREPARATION_BINDING, runDataGpu: prepareDispatch},
    {...CURATED_DRAINAGE_NETWORK_BINDING, runDataGpu: drainageDispatch}
  );
  expect(document.steps.find(step => step.id === 'drainage')!.inputs).toEqual({
    elevation: {stepId: 'prepare', output: 'prepared-surface'}
  });
  const plan = planWorkflow(document, {registry, host});
  expect(plan.steps['prepare'].issues.filter(issue => issue.severity === 'error')).toEqual([]);
  expect(plan.steps['drainage'].issues.filter(issue => issue.severity === 'error')).toEqual([]);
  expect(
    plan.steps['drainage'].issues.filter(issue => issue.code === 'capacity-may-exceed')
  ).toEqual([]);
  const runner = new WorkflowRunner({host, registry});
  try {
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.status, JSON.stringify(record)).toBe('done');
    expect(prepareDispatch).toHaveBeenCalledOnce();
    expect(drainageDispatch).toHaveBeenCalledOnce();
    const prepared = runner.getState().steps['prepare'].result!.outputs['prepared-surface'];
    expect(prepared.kind).toBe('raster');
    expect(prepared.table.geometryColumn).toBeUndefined();
    expect(prepared.raster?.grid).toEqual(elevation.grid);
    expect(prepared.raster?.cellSizeMode).toBe('uniform');
    expect(prepared.raster?.band.index).toBe(elevation.band.index);
    expect(elevation.band.values[4]).toBe(-999);
    expect(elevation.band.validity[4]).toBe(0);
  } finally {
    await runner.destroy();
  }
});
