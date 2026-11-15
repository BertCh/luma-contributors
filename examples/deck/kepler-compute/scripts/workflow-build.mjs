// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {
  addStepFromPalette,
  connectPorts,
  dock,
  dragHandle,
  frameGraph,
  getWorkbenchState,
  openDock,
  POLYGON_TEXT,
  runAcceptance,
  runAll,
  screenshot,
  setParameter,
  TEST_IDS,
  waitForWorkbench
} from './workflow-utils.mjs';

await runAcceptance('workflow-build', async (page, directory) => {
  assert.equal(
    (await getWorkbenchState(page)).dock.open,
    false,
    'Fresh app must start with the collapsed workbench bar'
  );
  await openDock(page);
  await dock(page).getByTestId(TEST_IDS.workflowNew).click();
  assert.equal(
    (await getWorkbenchState(page)).document.steps.length,
    0,
    'New workflow must have no steps'
  );
  await screenshot(page, directory, 'workflow-build-empty-dock');
  // Tab opens the palette after a pointer click on the canvas; keyboard focus keeps normal
  // Tab navigation.
  await dock(page)
    .getByTestId(TEST_IDS.canvas)
    .locator('.react-flow__pane')
    .click({position: {x: 12, y: 12}});
  await page.keyboard.press('Tab');
  await page.getByTestId(TEST_IDS.palette).waitFor();
  await page.getByTestId(TEST_IDS.paletteSearch).fill('geojson');
  await page.getByTestId(TEST_IDS.paletteItem('source-geojson')).waitFor();
  await screenshot(page, directory, 'workflow-build-palette-open');
  const sourceId = await addStepFromPalette(page, 'geojson', 'source-geojson');
  await setParameter(page, 'text', POLYGON_TEXT);
  const pointsId = await addStepFromPalette(
    page,
    'Random points in polygon',
    'random-points-in-polygon'
  );
  await connectPorts(page, sourceId, 'data', pointsId, 'polygons');
  let state = await getWorkbenchState(page);
  assert.deepEqual(
    state.document.steps.find(step => step.id === pointsId).inputs.polygons,
    {stepId: sourceId, output: 'data'},
    'Dragging source data to Polygons must create the edge'
  );

  await frameGraph(page);
  const emptyPoint = await dock(page)
    .getByTestId(TEST_IDS.canvas)
    .evaluate(canvas => {
      const bounds = canvas.getBoundingClientRect();
      for (const vertical of [0.8, 0.2, 0.6]) {
        for (const horizontal of [0.85, 0.5, 0.2]) {
          const point = {
            x: bounds.x + bounds.width * horizontal,
            y: bounds.y + bounds.height * vertical
          };
          if (document.elementFromPoint(point.x, point.y)?.classList.contains('react-flow__pane')) {
            return point;
          }
        }
      }
      throw new Error('Expected an empty React Flow pane location for the output drag');
    });
  await dragHandle(
    page,
    dock(page).getByTestId(TEST_IDS.outputPort(pointsId, 'random-points')),
    emptyPoint
  );
  await page.getByTestId(TEST_IDS.palette).waitFor();
  assert.deepEqual(
    (await getWorkbenchState(page)).palette.connectFrom,
    {stepId: pointsId, output: 'random-points'},
    'Output drag must open a connected palette'
  );
  assert.equal(
    await page.getByLabel('Compatible only', {exact: true}).isChecked(),
    true,
    'Output drag palette must default to compatible-only'
  );
  await page.getByTestId(TEST_IDS.paletteSearch).fill('random points in polygon');
  assert.equal(
    await page.getByTestId(TEST_IDS.paletteItem('random-points-in-polygon')).count(),
    0,
    'Points output must exclude the polygon-only random-points operation'
  );
  const voronoiId = await addStepFromPalette(page, 'Voronoi diagram', 'voronoi-diagram');
  state = await getWorkbenchState(page);
  assert.deepEqual(
    state.document.steps.find(step => step.id === voronoiId).inputs.points,
    {stepId: pointsId, output: 'random-points'},
    'Compatible palette must connect the new Voronoi'
  );
  await dock(page).getByTestId(TEST_IDS.node(pointsId)).click();
  await setParameter(page, 'pointsPerPolygon', 40);
  await frameGraph(page);
  await screenshot(page, directory, 'workflow-build-graph-before-run');
  state = await runAll(page);
  const expectedRows = {
    [sourceId]: ['data', 1],
    [pointsId]: ['random-points', 40],
    [voronoiId]: ['voronoi-cells', 40]
  };
  for (const [stepId, [output, count]] of Object.entries(expectedRows)) {
    assert.equal(
      state.steps[stepId].result.outputs[output].rowCount,
      count,
      `${stepId}.${output} must contain exactly ${count} rows`
    );
    assert.match(
      await dock(page).getByTestId(TEST_IDS.node(stepId)).innerText(),
      new RegExp(`\\b${count} rows\\b`),
      `${stepId} node summary must show ${count} rows`
    );
  }
  await screenshot(page, directory, 'workflow-build-graph-after-run');
  await dock(page).getByTestId(TEST_IDS.node(voronoiId)).click();
  await dock(page).getByTestId(TEST_IDS.inspectorTab('result')).click();
  const output = dock(page).getByTestId(TEST_IDS.resultOutput('voronoi-cells'));
  assert.match(await output.innerText(), /40 rows/, 'Result output must report 40 rows');
  const table = output.getByTestId(TEST_IDS.resultTable('voronoi-cells'));
  await table.waitFor();
  // The result table may paginate, but this fixture fits its first page.
  assert.equal(await table.locator('tbody tr').count(), 40, 'Result table must render 40 rows');
  await screenshot(page, directory, 'workflow-build-result-tab');
  await dock(page).getByTestId(TEST_IDS.nodeDisplay(voronoiId)).click();
  await waitForWorkbench(
    page,
    'Voronoi display must publish a dataset and visible map layer',
    stepId => {
      const step = window.__keplerWorkflowTest.store.getState().runner.steps[stepId];
      const published = step.published?.['voronoi-cells'];
      const computeTest = window.__keplerComputeTest;
      const datasets = computeTest.store?.getState().keplerGl.compute.visState.datasets;
      const datasetFound = datasets
        ? Object.values(datasets).some(dataset => /Voronoi/i.test(dataset.label))
        : Boolean(
            published &&
              /Voronoi/i.test(published.label) &&
              Object.keys(computeTest.getDatasetFieldTypes(published.datasetId)).length > 0
          );
      return (
        datasetFound &&
        computeTest
          .getState()
          .nativeLayers.some(
            layer => layer.dataId === published?.datasetId && layer.visible && layer.valid !== false
          )
      );
    },
    voronoiId,
    45000
  );
  // Current dev hooks do not expose the native store: also verify the Kepler Layers panel text.
  if (!(await page.evaluate(() => Boolean(window.__keplerComputeTest.store)))) {
    await page.locator('.side-panel__tab[data-for="layer-nav"]').click();
    await page
      .locator('.layer-manager')
      .getByText(/Voronoi/)
      .first()
      .waitFor();
  }
  await screenshot(page, directory, 'workflow-build-map-with-layer');
  return {
    rows: Object.fromEntries(
      Object.entries(expectedRows).map(([stepId, [, rows]]) => [stepId, rows])
    )
  };
});
