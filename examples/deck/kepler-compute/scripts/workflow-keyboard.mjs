// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {
  checkFrameTransition,
  dock,
  getWorkbenchState,
  openDock,
  POLYGON_TEXT,
  readCanvasFrame,
  runAcceptance,
  screenshot,
  TEST_IDS,
  waitForStepStatus,
  waitForWorkbench
} from './workflow-utils.mjs';

async function checkFocus(page, action, palette = false) {
  const focus = await page.evaluate(
    ({dockId, paletteId}) => {
      const element = document.activeElement;
      return {
        tag: element?.tagName,
        testId: element?.getAttribute('data-testid'),
        insideDock: Boolean(element?.closest(`[data-testid="${dockId}"]`)),
        insidePalette: Boolean(element?.closest(`[data-testid="${paletteId}"]`))
      };
    },
    {dockId: TEST_IDS.dock, paletteId: TEST_IDS.palette}
  );
  assert.equal(
    focus.insideDock,
    true,
    `${action}: focus must remain inside the dock; saw ${JSON.stringify(focus)}`
  );
  assert.equal(
    palette ? focus.insidePalette : focus.insideDock,
    true,
    `${action}: focus must stay in ${palette ? 'the palette' : 'the dock'}; saw ${JSON.stringify(focus)}`
  );
}

async function focusCanvasFromToolbar(page) {
  await dock(page).getByTestId(TEST_IDS.toolbar).getByTestId(TEST_IDS.dockToggle).focus();
  await checkFocus(page, 'Toolbar focus');
  await page.keyboard.press('Tab');
  const focus = await dock(page)
    .getByTestId(TEST_IDS.canvas)
    .evaluate(element => ({
      focused: document.activeElement === element,
      tag: document.activeElement?.tagName,
      testId: document.activeElement?.getAttribute('data-testid')
    }));
  assert.equal(
    focus.focused,
    true,
    `Tab from the last toolbar control must focus the canvas; saw ${JSON.stringify(focus)}`
  );
  assert.equal(
    (await getWorkbenchState(page)).palette.open,
    false,
    'Tab from the toolbar must focus the canvas without opening the palette'
  );
  await checkFocus(page, 'Tab focuses canvas from toolbar');
}

await runAcceptance('workflow-keyboard', async (page, directory) => {
  await openDock(page);
  // Setup is outside this scenario's keyboard-only portion.
  await page.evaluate(text => {
    const store = window.__keplerWorkflowTest.store;
    store.createWorkflow({title: 'Keyboard acceptance'});
    store.edit('keyboard-fixture', document => ({
      ...document,
      steps: [
        {
          id: 'source',
          operationId: 'source-geojson',
          inputs: {},
          parameters: {text: {from: 'static', value: text}},
          position: {x: 0, y: 0}
        }
      ]
    }));
    store.select(['source']);
  }, POLYGON_TEXT);
  const canvas = dock(page).getByTestId(TEST_IDS.canvas);
  await focusCanvasFromToolbar(page);
  // React Flow exposes node controls before the inspector in the normal tab sequence.
  const tabStops = [];
  let reachedInspector = false;
  for (let index = 0; index < 50 && !reachedInspector; index++) {
    await page.keyboard.press('Tab');
    const focus = await page.evaluate(
      ({canvasId, inspectorId}) => {
        const element = document.activeElement;
        return {
          tag: element?.tagName,
          testId: element?.getAttribute('data-testid'),
          insideCanvas: Boolean(element?.closest(`[data-testid="${canvasId}"]`)),
          insideInspector: Boolean(element?.closest(`[data-testid="${inspectorId}"]`)),
          paletteOpen: window.__keplerWorkflowTest.store.getState().palette.open
        };
      },
      {canvasId: TEST_IDS.canvas, inspectorId: TEST_IDS.inspector}
    );
    tabStops.push(focus);
    assert.equal(
      focus.paletteOpen,
      false,
      `Tab from keyboard-focused canvas must not open the palette; visited ${JSON.stringify(tabStops)}`
    );
    assert.equal(
      focus.insideCanvas || focus.insideInspector,
      true,
      `Tab must advance through canvas controls to the inspector; visited ${JSON.stringify(tabStops)}`
    );
    await checkFocus(page, 'Tab advances toward inspector');
    reachedInspector = focus.insideInspector;
  }
  assert.equal(
    reachedInspector,
    true,
    `Keyboard Tab must reach the inspector within 50 stops; visited ${JSON.stringify(tabStops)}`
  );
  await screenshot(page, directory, 'workflow-keyboard-tab-to-inspector');
  await focusCanvasFromToolbar(page);
  await page.keyboard.press('n');
  await page.getByTestId(TEST_IDS.paletteSearch).waitFor();
  await checkFocus(page, 'N opens palette', true);
  await page.keyboard.type('source-geojson');
  await page.keyboard.press('ArrowDown');
  await checkFocus(page, 'ArrowDown in palette', true);
  const highlighted = await page
    .getByTestId(TEST_IDS.paletteSearch)
    .getAttribute('aria-activedescendant');
  assert.equal(typeof highlighted, 'string', 'ArrowDown must highlight a palette operation');
  const expectedOperationId = await page
    .locator(`[id="${highlighted}"]`)
    .getAttribute('data-testid');
  assert.equal(
    expectedOperationId,
    TEST_IDS.paletteItem('source-geojson'),
    'GeoJSON search must highlight the GeoJSON source'
  );
  await page.keyboard.press('Enter');
  await page.getByTestId(TEST_IDS.palette).waitFor({state: 'hidden'});
  let state = await getWorkbenchState(page);
  assert.equal(state.document.steps.length, 2, 'Enter must add one step');
  const addedId = state.selectedStepIds[0];
  assert.equal(
    state.document.steps.find(step => step.id === addedId)?.operationId,
    'source-geojson',
    'Enter must add the highlighted operation and select it'
  );
  await checkFocus(page, 'Enter adds step');
  // Move the existing source as fixture setup. It has no pending new-node reveal, so only F
  // can bring it back; merely asserting that the newly added node is visible misses a no-op.
  await page.evaluate(() => {
    const store = window.__keplerWorkflowTest.store;
    store.select(['source']);
    store.edit('offscreen-frame-fixture', document => ({
      ...document,
      steps: document.steps.map(step =>
        step.id === 'source' ? {...step, position: {x: 5000, y: 5000}} : step
      )
    }));
  });
  await waitForWorkbench(
    page,
    'F fixture source must start outside the canvas',
    ({nodeId, canvasId}) => {
      const node = document.querySelector(`[data-testid="${nodeId}"]`)?.getBoundingClientRect();
      const canvas = document.querySelector(`[data-testid="${canvasId}"]`)?.getBoundingClientRect();
      return node && canvas && (node.left > canvas.right || node.top > canvas.bottom);
    },
    {canvasId: TEST_IDS.canvas, nodeId: TEST_IDS.node('source')}
  );
  const beforeFrame = await readCanvasFrame(page, 'source');
  await page.keyboard.press('f');
  await checkFocus(page, 'F frames selected node');
  await waitForWorkbench(
    page,
    'F must frame the complete selected node inside the canvas',
    ({canvasId, nodeId}) => {
      const canvasBounds = document
        .querySelector(`[data-testid="${canvasId}"]`)
        ?.getBoundingClientRect();
      const nodeBounds = document
        .querySelector(`[data-testid="${nodeId}"]`)
        ?.getBoundingClientRect();
      return (
        canvasBounds &&
        nodeBounds &&
        nodeBounds.left >= canvasBounds.left &&
        nodeBounds.right <= canvasBounds.right &&
        nodeBounds.top >= canvasBounds.top &&
        nodeBounds.bottom <= canvasBounds.bottom
      );
    },
    {canvasId: TEST_IDS.canvas, nodeId: TEST_IDS.node('source')}
  );
  checkFrameTransition(beforeFrame, await readCanvasFrame(page, 'source'));
  await screenshot(page, directory, 'workflow-keyboard-framed');
  await page.evaluate(stepId => window.__keplerWorkflowTest.store.select([stepId]), addedId);
  await page.keyboard.press('f');
  await page.keyboard.press('b');
  await waitForStepStatus(page, addedId, 'disabled');
  await checkFocus(page, 'B disables');
  await page.keyboard.press('d');
  state = await getWorkbenchState(page);
  assert.deepEqual(
    state.document.steps.find(step => step.id === addedId).display,
    ['data'],
    'D must toggle display on the selected step'
  );
  await checkFocus(page, 'D toggles display');
  const modifier = await page.evaluate(() =>
    /mac|iphone|ipad/i.test(navigator.platform) ? 'Meta' : 'Control'
  );
  await page.keyboard.press(`${modifier}+z`);
  assert.equal(
    (await getWorkbenchState(page)).document.steps
      .find(step => step.id === addedId)
      .display?.includes('data') ?? false,
    false,
    'Mod+Z must undo display'
  );
  await checkFocus(page, 'Mod+Z undoes');
  await page.keyboard.press(`${modifier}+Shift+z`);
  assert.deepEqual(
    (await getWorkbenchState(page)).document.steps.find(step => step.id === addedId).display,
    ['data'],
    'Shift+Mod+Z must redo display'
  );
  await checkFocus(page, 'Shift+Mod+Z redoes');
  await screenshot(page, directory, 'workflow-keyboard-redo');
  await page.keyboard.press('Delete');
  assert.deepEqual(
    (await getWorkbenchState(page)).document.steps.map(step => step.id),
    ['source'],
    'Delete must remove only the selected step'
  );
  await checkFocus(page, 'Delete removes');
  await page.keyboard.press('Shift+Enter');
  await waitForWorkbench(page, 'Shift+Enter must finish a whole-workflow run', () => {
    const state = window.__keplerWorkflowTest.store.getState();
    return !state.runner.running && state.runLog[0]?.status === 'done' && !state.runLog[0].target;
  });
  await waitForStepStatus(page, 'source', 'done');
  await checkFocus(page, 'Shift+Enter runs');
  await screenshot(page, directory, 'workflow-keyboard-run');
  await page.keyboard.press('n');
  await page.getByTestId(TEST_IDS.paletteSearch).waitFor();
  await checkFocus(page, 'N reopens palette', true);
  await page.keyboard.press('Escape');
  await page.getByTestId(TEST_IDS.palette).waitFor({state: 'hidden'});
  await checkFocus(page, 'Escape closes palette');

  await page.keyboard.press(`${modifier}+a`);
  assert.deepEqual(
    (await getWorkbenchState(page)).selectedStepIds,
    ['source'],
    'Mod+A must select the remaining source'
  );
  await checkFocus(page, 'Mod+A selects source');
  // Tab-order coverage lives in workflow-accessibility; here isolate text-field shortcut handling.
  await dock(page).getByTestId(TEST_IDS.stepLabel).focus();
  await checkFocus(page, 'Text field focus');
  await page.keyboard.press('Tab');
  assert.equal(
    (await getWorkbenchState(page)).palette.open,
    false,
    'Tab inside the step label field must retain normal text-field navigation'
  );
  await checkFocus(page, 'Tab leaves text field');
  await screenshot(page, directory, 'workflow-keyboard-text-field-tab');
  // This separate pointer check preserves Tab-to-add after a canvas click. The sequence above
  // uses only keyboard input after fixture setup; N is its unconditional add-step shortcut.
  await canvas.locator('.react-flow__pane').click({position: {x: 12, y: 12}});
  await page.keyboard.press('n');
  await page.getByTestId(TEST_IDS.paletteSearch).waitFor();
  await checkFocus(page, 'N after a canvas click opens palette', true);
  await screenshot(page, directory, 'workflow-keyboard-pointer-n');
  await page.keyboard.press('Escape');
  await page.getByTestId(TEST_IDS.palette).waitFor({state: 'hidden'});
  await checkFocus(page, 'Escape closes pointer-N palette');
  await canvas.locator('.react-flow__pane').click({position: {x: 12, y: 12}});
  await page.keyboard.press('Tab');
  await page.getByTestId(TEST_IDS.paletteSearch).waitFor();
  await checkFocus(page, 'Tab after a canvas click opens palette', true);
  await screenshot(page, directory, 'workflow-keyboard-pointer-tab');
  await page.keyboard.press('Escape');
  await page.getByTestId(TEST_IDS.palette).waitFor({state: 'hidden'});
  await checkFocus(page, 'Escape closes pointer-opened palette');
  // AX-60: prepare actual user data through the gallery, then use only three typed
  // characters and Enter. No test host/profile override stands in for dataset scoping.
  await page.getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.getByText('Prepare data', {exact: true}).click();
  await page.getByText('Try a sample dataset', {exact: true}).click();
  const gallerySelect = page.getByTestId('ax-original-sample-select');
  await gallerySelect.selectOption('points-time');
  const sampleTitle = await gallerySelect.locator('option[value="points-time"]').innerText();
  await page.getByRole('button', {name: 'Prepare sample', exact: true}).click();
  await page.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await page.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.locator('.ReactModal__Overlay').waitFor({state: 'detached', timeout: 15000});
  await waitForWorkbench(
    page,
    'AX palette requires the published point dataset profile',
    title => {
      const host = window.__keplerWorkflowTest.store.host;
      return host
        .listDatasets()
        .some(dataset => dataset.label === title && host.getAnalysisProfile(dataset.id));
    },
    sampleTitle
  );
  const activeDatasetId = await page.evaluate(
    title =>
      window.__keplerWorkflowTest.store.host.listDatasets().find(dataset => dataset.label === title)
        .id,
    sampleTitle
  );
  await page.evaluate(datasetId => {
    const store = window.__keplerWorkflowTest.store;
    store.createWorkflow({title: 'AX promoted dataset scope'});
    store.edit('ax-promoted-source', document => ({
      ...document,
      inputs: [{id: 'source-id', label: 'Source', kind: 'string', value: datasetId}],
      steps: [
        {
          id: 'source',
          operationId: 'source-dataset',
          inputs: {},
          parameters: {dataset: {from: 'input', inputId: 'source-id'}},
          position: {x: 0, y: 0}
        }
      ]
    }));
    store.select(['source']);
    store.setDock({open: false});
  }, activeDatasetId);
  await dock(page).getByTestId(TEST_IDS.dockToggle).focus();
  await page.keyboard.press(`${modifier}+k`);
  const commandPalette = page.getByTestId('ax-command-palette');
  await commandPalette.waitFor();
  const commandDataset = page.getByTestId('ax-command-dataset');
  assert.equal(
    await commandDataset.inputValue(),
    activeDatasetId,
    'Palette scope must resolve the selected source workflow input without manually switching datasets'
  );
  const commandSearch = page.getByTestId(TEST_IDS.paletteSearch);
  await commandSearch.focus();
  await page.keyboard.type('hot');
  const commandHighlight = await commandSearch.getAttribute('aria-activedescendant');
  assert.equal(
    await page.locator(`[id="${commandHighlight}"]`).getAttribute('data-testid'),
    'ax-command-template:hot-spot-map',
    'hot must highlight the mapped hot spot template'
  );
  await checkFocus(page, 'Cmd/Ctrl-K search focus', true);
  await page.keyboard.press('Enter');
  await page.getByTestId('ax-analysis-mapping').waitFor();
  assert.equal(
    await page.getByTestId('ax-mapping-dataset-points').inputValue(),
    activeDatasetId,
    'Enter must open the hot spot template with the active dataset slot filled'
  );
  assert.match(
    await page.getByTestId('ax-mapping-progress').innerText(),
    /1 of 1 mapped/,
    'The hot spot template must open with every required slot filled'
  );
  await waitForWorkbench(page, 'Enter focuses the template mapping', () =>
    Boolean(document.activeElement?.closest('[data-testid="ax-analysis-mapping"]'))
  );
  await screenshot(page, directory, 'workflow-keyboard-hot-template-mapped');
  await page.keyboard.press('Escape');
  await commandPalette.waitFor({state: 'hidden'});
  await checkFocus(page, 'Escape leaves AX palette');
  // The capture listener must work even when search itself stops bubbling keyboard events.
  await page.keyboard.press(`${modifier}+k`);
  await commandPalette.waitFor();
  await commandSearch.focus();
  await page.keyboard.press(`${modifier}+k`);
  await commandPalette.waitFor({state: 'hidden'});
  return {
    addedId,
    modifier,
    axCommandPalette: {query: 'hot', activeDatasetId, slotsMapped: true, keyboardFocus: true},
    keyboardTabMovesToInspector: true,
    pointerTabOpensPalette: true,
    paletteOpensWithN: {keyboard: true, pointer: true}
  };
});
