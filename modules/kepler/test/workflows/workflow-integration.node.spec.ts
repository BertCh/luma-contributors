// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {createWorkflowDocument} from '../../src/workflows/workflow-document';
import {
  addWorkflowStep,
  promoteWorkflowParameter,
  setWorkflowParameter,
  updateWorkflowInput,
  updateWorkflowStep
} from '../../src/workflows/workflow-edit';
import {exportWorkflowJSON, importWorkflowJSON} from '../../src/workflows/workflow-export';
import {WorkflowHistory} from '../../src/workflows/workflow-history';
import {getWorkflowOperationRegistry} from '../../src/workflows/workflow-operations';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import type {WorkflowDocument} from '../../src/workflows/workflow-types';
import {makeFixtureHost} from './workflow-fixtures';

// The whole library on real catalog operations, without a GPU: a density surface, hot spots on
// it, and the statistics joined back to the cells so they can be mapped.

const EXTENT = JSON.stringify({
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      id: 'district',
      properties: {name: 'District'},
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [-87.7, 41.8],
            [-87.66, 41.8],
            [-87.66, 41.83],
            [-87.7, 41.83],
            [-87.7, 41.8]
          ]
        ]
      }
    }
  ]
});

function buildDocument(): WorkflowDocument {
  const registry = getWorkflowOperationRegistry();
  let document = createWorkflowDocument({title: 'Density hot spots'});
  const add = (
    operationId: string,
    parameters: Record<string, string | number | boolean>,
    connectFrom?: {stepId: string; output: string},
    connectTo?: string
  ) => {
    const added = addWorkflowStep(document, registry, {
      operationId,
      parameters,
      connectFrom,
      connectTo
    });
    document = added.document;
    return added.stepId;
  };
  const extent = add('source-geojson', {text: EXTENT});
  const points = add(
    'random-points-in-polygon',
    {pointsPerPolygon: 200, seed: 7},
    {stepId: extent, output: 'data'}
  );
  const density = add(
    'kernel-density-grid',
    {cellSize: 250, bandwidth: 600},
    {stepId: points, output: 'random-points'}
  );
  const hotspots = add(
    'hotspot-gi-star',
    {valueProperty: 'density', neighborCount: 8},
    {stepId: density, output: 'density-cells'}
  );
  const joined = add(
    'table-join-attributes',
    {targetKey: 'cell_id'},
    {stepId: density, output: 'density-cells'},
    'target'
  );
  document = {
    ...document,
    steps: document.steps.map(step =>
      step.id === joined
        ? {...step, inputs: {...step.inputs, attributes: {stepId: hotspots, output: 'hotspots'}}}
        : step
    )
  };
  return updateWorkflowStep(document, joined, {display: ['joined']});
}

describe('workflow integration on catalog operations', () => {
  test('runs a five-step chain, re-runs only what an edit invalidates, and restores by undo', async () => {
    const registry = getWorkflowOperationRegistry();
    const host = makeFixtureHost();
    const runner = new WorkflowRunner({host, registry});
    const document = buildDocument();
    const history = new WorkflowHistory(document);
    runner.setDocument(document);

    const planned = runner.getState();
    expect(planned.issues.filter(issue => issue.severity === 'error')).toEqual([]);
    expect(planned.order).toHaveLength(5);
    for (const stepId of planned.order)
      expect([stepId, planned.steps[stepId].status]).toEqual([stepId, 'ready']);

    const first = await runner.run();
    expect(first.steps.map(step => [step.stepId, step.status])).toEqual(
      planned.order.map(stepId => [stepId, 'done'])
    );
    const state = runner.getState();
    const [extent, points, density, hotspots, joined] = planned.order;
    expect(state.steps[points].result?.outputs['random-points'].table.rowCount).toBe(200);
    const cells = state.steps[density].result?.outputs['density-cells'];
    const joinedData = state.steps[joined].result?.outputs['joined'];
    expect(cells && cells.table.rowCount > 0 && cells.table.rowCount <= 3000).toBe(true);
    expect(joinedData?.table.rowCount).toBe(cells?.table.rowCount);
    expect(joinedData?.kind).toBe('polygon');
    // The Gi* statistics arrived on the cells.
    const hotspotColumns = Object.keys(
      state.steps[hotspots].result?.outputs['hotspots'].table.columnTypes ?? {}
    ).filter(name => name !== 'source_id' && name !== 'source_index');
    expect(hotspotColumns.length).toBeGreaterThan(0);
    for (const name of hotspotColumns)
      expect(Object.keys(joinedData?.table.columnTypes ?? {})).toContain(name);
    expect(host.published.map(entry => [entry.stepId, entry.output])).toEqual([[joined, 'joined']]);

    // Changing the neighbour count re-keys Gi* and the join only.
    const edited = setWorkflowParameter(document, hotspots, 'neighborCount', {
      from: 'static',
      value: 4
    });
    history.apply(edited, 'parameter');
    runner.setDocument(edited);
    const afterEdit = runner.getState();
    expect(
      [extent, points, density, hotspots, joined].map(stepId => afterEdit.steps[stepId].status)
    ).toEqual(['done', 'done', 'done', 'stale', 'stale']);
    const second = await runner.run();
    expect(second.steps.map(step => step.stepId)).toEqual([hotspots, joined]);
    expect(host.published).toHaveLength(2);
    expect(host.published[1].datasetId).toBe(host.published[0].datasetId);

    // Undo: the earlier results are still cached, so nothing runs.
    expect(history.undo()).toBe(true);
    runner.setDocument(history.document);
    const afterUndo = runner.getState();
    for (const stepId of afterUndo.order) expect(afterUndo.steps[stepId].status).toBe('done');

    await runner.destroy();
  });

  test('refuses a hand-off over the consumer capacity and names the limit', async () => {
    const registry = getWorkflowOperationRegistry();
    const runner = new WorkflowRunner({host: makeFixtureHost(), registry});
    const document = buildDocument();
    const density = document.steps.find(step => step.operationId === 'kernel-density-grid');
    const hotspots = document.steps.find(step => step.operationId === 'hotspot-gi-star');
    // 30 m cells over a 3.3 km by 3.3 km extent are far more than the 3,000 features Gi* accepts.
    runner.setDocument(
      setWorkflowParameter(document, density?.id ?? '', 'cellSize', {from: 'static', value: 30})
    );
    const record = await runner.run();
    expect(record.status).toBe('error');
    const failure = runner.getState().steps[hotspots?.id ?? ''];
    expect(failure.status).toBe('error');
    expect(failure.error?.code).toBe('capacity-exceeded');
    expect(failure.error?.message).toMatch(/3,?000/);
    await runner.destroy();
  });

  test('a workflow input drives a promoted parameter and survives a JSON round trip', async () => {
    const registry = getWorkflowOperationRegistry();
    const document = buildDocument();
    const points = document.steps.find(step => step.operationId === 'random-points-in-polygon');
    const promoted = promoteWorkflowParameter(
      document,
      registry,
      points?.id ?? '',
      'pointsPerPolygon'
    );
    const changed = updateWorkflowInput(promoted.document, promoted.inputId, {value: 50});
    const reloaded = importWorkflowJSON(exportWorkflowJSON(changed));
    expect(reloaded).toEqual(changed);
    const runner = new WorkflowRunner({host: makeFixtureHost(), registry});
    runner.setDocument(reloaded);
    await runner.run({target: points?.id});
    expect(
      runner.getState().steps[points?.id ?? ''].result?.outputs['random-points'].table.rowCount
    ).toBe(50);
    await runner.destroy();
  });
});
