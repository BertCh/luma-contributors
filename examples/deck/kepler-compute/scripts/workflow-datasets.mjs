// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {
  addStepFromPalette,
  connectPorts,
  dock,
  frameGraph,
  getWorkbenchState,
  openDock,
  runAcceptance,
  runAll,
  screenshot,
  setParameter,
  TEST_IDS,
  waitForStepStatus,
  waitForWorkbench
} from './workflow-utils.mjs';

await runAcceptance('workflow-datasets', async (page, directory) => {
  await openDock(page);
  await page.evaluate(() =>
    window.__keplerWorkflowTest.store.createWorkflow({title: 'Selected observations'})
  );
  const dataset = await page.evaluate(() =>
    window.__keplerWorkflowTest.store.host
      .listDatasets()
      .find(item => item.id === 'resident-observations')
  );
  assert.notEqual(dataset, undefined, 'Host must offer the built-in resident-observations source');
  assert.equal(dataset.resident, true, 'Built-in observations must be GPU resident');
  const sourceId = await addStepFromPalette(page, 'Dataset', 'source-dataset');
  await setParameter(page, 'dataset', dataset.id);
  await setParameter(page, 'population', 'selected');
  const pointsId = await addStepFromPalette(page, 'Voronoi diagram', 'voronoi-diagram');
  await connectPorts(page, sourceId, 'data', pointsId, 'points');
  const before = await runAll(page);
  const initialRows = before.steps[sourceId].result.outputs.data.rowCount;
  assert.equal(initialRows, 256, 'Rows: selected must initially read all 256 observations');
  await screenshot(page, directory, 'workflow-datasets-all-selected');
  const expectedRows = await page.evaluate(() => {
    const computeTest = window.__keplerComputeTest;
    const expected = computeTest.source().filter(row => row[2] >= 40 && row[2] <= 75).length;
    computeTest.setRange(40, 75);
    return expected;
  });
  assert.equal(
    expectedRows > 0 && expectedRows < initialRows,
    true,
    `Fixture range 40–75 must narrow the selection; got ${expectedRows} of ${initialRows}`
  );
  await waitForWorkbench(
    page,
    'Native range filter must update selectedCount',
    count => window.__keplerComputeTest.getState().selectedCount === count,
    expectedRows
  );
  await waitForStepStatus(page, sourceId, 'stale');
  await waitForStepStatus(page, pointsId, 'stale');
  await screenshot(page, directory, 'workflow-datasets-selection-stale');
  const after = await runAll(page);
  assert.equal(
    after.steps[sourceId].result.outputs.data.rowCount,
    expectedRows,
    'Re-run source count must equal the native filtered selection'
  );
  assert.equal(
    after.steps[pointsId].result.outputs['voronoi-cells'].rowCount,
    expectedRows,
    'Point operation must consume the new selection'
  );
  assert.notEqual(
    after.steps[sourceId].key,
    before.steps[sourceId].key,
    'External selection change must change the source content key'
  );
  await frameGraph(page);
  await screenshot(page, directory, 'workflow-datasets-filtered-done');
  await dock(page).getByTestId(TEST_IDS.nodeDisplay(pointsId)).click();
  await waitForWorkbench(
    page,
    'Displayed output must publish a visible map layer',
    stepId => {
      const published =
        window.__keplerWorkflowTest.store.getState().runner.steps[stepId].published?.[
          'voronoi-cells'
        ];
      return Boolean(
        published &&
          window.__keplerComputeTest
            .getState()
            .nativeLayers.some(
              layer =>
                layer.dataId === published.datasetId && layer.visible && layer.valid !== false
            )
      );
    },
    pointsId,
    45000
  );
  const published = (await getWorkbenchState(page)).steps[pointsId].published['voronoi-cells'];
  await dock(page).getByTestId(TEST_IDS.node(sourceId)).click();
  await dock(page).getByTestId(TEST_IDS.inspectorTab('parameters')).click();
  const offered = await dock(page)
    .getByTestId(TEST_IDS.parameter('dataset'))
    .locator('option')
    .evaluateAll(elements =>
      elements.map(element => ({value: element.value, label: element.textContent}))
    );
  assert.equal(
    offered.some(option => option.value === dataset.id),
    true,
    `Dataset selector must still offer observations; saw ${JSON.stringify(offered)}`
  );
  assert.equal(
    offered.some(option => option.value === published.datasetId),
    false,
    `Dataset selector must exclude its published workflow output ${published.datasetId}; saw ${JSON.stringify(offered)}`
  );
  const hostIds = await page.evaluate(() =>
    window.__keplerWorkflowTest.store.host.listDatasets().map(item => item.id)
  );
  assert.equal(
    hostIds.includes(published.datasetId),
    false,
    'Host dataset list must prevent workflow-output feedback loops'
  );
  await screenshot(page, directory, 'workflow-datasets-published-excluded');
  return {initialRows, filteredRows: expectedRows, publishedDatasetId: published.datasetId};
});
