// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

// Mirrored from src/workflow/workbench-contract.ts; Node cannot import that TypeScript module.
export const TEST_IDS = {
  dock: 'workflow-dock',
  dockToggle: 'workflow-dock-toggle',
  toolbar: 'workflow-toolbar',
  workflowNew: 'workflow-new',
  runAll: 'workflow-run-all',
  runToHere: 'workflow-run-to-here',
  undo: 'workflow-undo',
  redo: 'workflow-redo',
  addStep: 'workflow-add-step',
  templates: 'workflow-templates',
  template: templateId => `workflow-template-${templateId}`,
  canvas: 'workflow-canvas',
  node: stepId => `workflow-node-${stepId}`,
  nodeStatus: stepId => `workflow-node-status-${stepId}`,
  nodeDisplay: stepId => `workflow-node-display-${stepId}`,
  nodeDisable: stepId => `workflow-node-disable-${stepId}`,
  inputPort: (stepId, port) => `workflow-port-in-${stepId}-${port}`,
  outputPort: (stepId, port) => `workflow-port-out-${stepId}-${port}`,
  palette: 'workflow-palette',
  paletteSearch: 'workflow-palette-search',
  paletteItem: operationId => `workflow-palette-item-${operationId}`,
  inspector: 'workflow-inspector',
  inspectorTab: tab => `workflow-inspector-tab-${tab}`,
  stepLabel: 'workflow-step-label',
  parameter: name => `workflow-parameter-${name}`,
  parameterError: name => `workflow-parameter-error-${name}`,
  inputBinding: port => `workflow-input-${port}`,
  result: 'workflow-result',
  resultOutput: output => `workflow-result-output-${output}`,
  resultTable: output => `workflow-result-table-${output}`,
  runLog: 'workflow-run-log',
  runLogToggle: 'workflow-run-log-toggle',
  sidePanel: 'workflow-side-panel'
};

export const POLYGON_TEXT = JSON.stringify({
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      id: 'acceptance-polygon',
      properties: {name: 'Acceptance polygon'},
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [-87.64, 41.88],
            [-87.62, 41.88],
            [-87.62, 41.9],
            [-87.64, 41.9],
            [-87.64, 41.88]
          ]
        ]
      }
    }
  ]
});

export function dock(page) {
  return page.getByTestId(TEST_IDS.dock);
}

export async function launchWorkbench(url = 'http://127.0.0.1:5580/?rows=256') {
  const browser = await chromium.launch(
    getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
  );
  const failures = [];
  const watchdog = setTimeout(() => {
    failures.push('Acceptance script exceeded the 180-second watchdog.');
    process.exitCode = 1;
    console.error(failures.at(-1));
    void browser.close();
  }, 180000);
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.setDefaultTimeout(15000);
    page.on('pageerror', error => failures.push(`pageerror: ${error.message}`));
    page.on('console', message => {
      if (
        message.type() === 'error' &&
        /WGSL|Invalid.*(?:Pipeline|Buffer)|validation(?: error)?/i.test(message.text())
      )
        failures.push(`console: ${message.text()}`);
    });
    await pinViteModules(page);
    await page.goto(url, {waitUntil: 'domcontentloaded'});
    await waitForWorkbench(
      page,
      'App selectedCount must become 256',
      () => window.__keplerComputeTest?.getState().selectedCount === 256,
      undefined,
      60000
    );
    await waitForWorkbench(page, 'Dev workbench store must be exposed', () =>
      Boolean(window.__keplerWorkflowTest?.store)
    );
    return {browser, page, failures, watchdog};
  } catch (error) {
    clearTimeout(watchdog);
    await browser.close();
    throw error;
  }
}

/** Keep result tables and GPU handles out of failure reports and Playwright serialization. */
export async function getWorkbenchState(page) {
  return page.evaluate(() => {
    const state = window.__keplerWorkflowTest?.store.getState();
    if (!state) return null;
    return {
      workflows: state.workflows,
      activeWorkflowId: state.activeWorkflowId,
      document: state.document,
      steps: Object.fromEntries(
        Object.entries(state.runner?.steps ?? {}).map(([stepId, step]) => [
          stepId,
          {
            status: step.status,
            key: step.key,
            issues: step.issues,
            error: step.error,
            published: step.published,
            result: step.result
              ? {
                  key: step.result.key,
                  engine: step.result.engine,
                  summary: step.result.summary,
                  durationMilliseconds: step.result.durationMilliseconds,
                  completedAt: step.result.completedAt,
                  outputs: Object.fromEntries(
                    Object.entries(step.result.outputs).map(([output, data]) => [
                      output,
                      {kind: data.kind, rowCount: data.table.rowCount}
                    ])
                  )
                }
              : null
          }
        ])
      ),
      running: state.runner?.running ?? false,
      selectedStepIds: state.selectedStepIds,
      notice: state.notice,
      dock: state.dock,
      palette: state.palette,
      dialog: state.dialog,
      runLog: state.runLog
    };
  });
}

export async function waitForWorkbench(page, expectation, predicate, argument, timeout = 30000) {
  try {
    await page.waitForFunction(predicate, argument, {timeout});
  } catch (error) {
    const state = await getWorkbenchState(page);
    throw new Error(`${expectation}. Observed: ${JSON.stringify(state)}`, {cause: error});
  }
}

export async function openDock(page) {
  if (!(await getWorkbenchState(page)).dock.open) {
    await dock(page).getByTestId(TEST_IDS.dockToggle).click();
  }
  await dock(page).getByTestId(TEST_IDS.toolbar).waitFor();
  assert.equal((await getWorkbenchState(page)).dock.open, true, 'Collapsed bar must open the dock');
}

export async function waitForStepStatus(page, stepId, status, timeout = 30000) {
  await waitForWorkbench(
    page,
    `Step ${stepId} must have node data-status="${status}"`,
    ({testId, expected}) => {
      const element = document.querySelector(`[data-testid="${testId}"]`);
      const step = window.__keplerWorkflowTest.store.getState().runner?.steps;
      const failure = Object.entries(step ?? {}).find(([, value]) => value.status === 'error');
      if (expected === 'done' && failure) {
        throw new Error(`${failure[0]}: ${failure[1].error?.message ?? 'Step failed'}`);
      }
      return element?.getAttribute('data-status') === expected;
    },
    {testId: TEST_IDS.node(stepId), expected: status},
    timeout
  );
}

export async function waitForRunIdle(page) {
  await waitForWorkbench(page, 'Workflow run must be idle', () => {
    const state = window.__keplerWorkflowTest.store.getState();
    return Boolean(state.runner && !state.runner.running);
  });
}

export async function addStepFromPalette(page, query, operationId) {
  if (!(await getWorkbenchState(page)).palette.open) {
    await dock(page).getByTestId(TEST_IDS.addStep).click();
  }
  const before = (await getWorkbenchState(page)).document.steps.map(step => step.id);
  await page.getByTestId(TEST_IDS.paletteSearch).fill(query);
  await page.getByTestId(TEST_IDS.paletteItem(operationId)).click();
  await waitForWorkbench(
    page,
    `Palette must add exactly one ${operationId} step`,
    ({previous, operation}) => {
      const state = window.__keplerWorkflowTest.store.getState();
      const added = state.document.steps.filter(step => !previous.includes(step.id));
      return !state.palette.open && added.length === 1 && added[0].operationId === operation;
    },
    {previous: before, operation: operationId}
  );
  return (await getWorkbenchState(page)).document.steps.find(step => !before.includes(step.id)).id;
}

export async function frameGraph(page) {
  await page.evaluate(() => window.__keplerWorkflowTest.store.select([]));
  const canvas = dock(page).getByTestId(TEST_IDS.canvas);
  await canvas.focus();
  await page.keyboard.press('f');
  // Wait out React Flow's documented 200 ms fit animation with browser animation frames.
  await page.evaluate(
    () =>
      new Promise(resolve => {
        const start = performance.now();
        function next() {
          if (performance.now() - start >= 350) resolve();
          else requestAnimationFrame(next);
        }
        requestAnimationFrame(next);
      })
  );
}

export async function readCanvasFrame(page, stepId) {
  return page.evaluate(
    ({canvasId, nodeId}) => {
      const canvas = document.querySelector(`[data-testid="${canvasId}"]`);
      const node = document.querySelector(`[data-testid="${nodeId}"]`);
      if (!canvas || !node) throw new Error(`Expected canvas ${canvasId} and node ${nodeId}`);
      const readBounds = element => {
        const {left, right, top, bottom} = element.getBoundingClientRect();
        return {left, right, top, bottom};
      };
      return {
        canvas: readBounds(canvas),
        node: readBounds(node),
        transform: canvas.querySelector('.react-flow__viewport')?.style.transform ?? ''
      };
    },
    {canvasId: TEST_IDS.canvas, nodeId: TEST_IDS.node(stepId)}
  );
}

export function checkFrameTransition(before, after) {
  const containsNode = frame =>
    frame.node.left >= frame.canvas.left &&
    frame.node.right <= frame.canvas.right &&
    frame.node.top >= frame.canvas.top &&
    frame.node.bottom <= frame.canvas.bottom;
  assert.equal(
    containsNode(before),
    false,
    `F fixture must start outside the canvas; observed ${JSON.stringify(before)}`
  );
  assert.equal(
    containsNode(after),
    true,
    `F must bring the complete node into view; observed ${JSON.stringify(after)}`
  );
  assert.notEqual(
    after.transform,
    before.transform,
    `F must change the viewport transform; observed ${after.transform}`
  );
}

async function checkHandleHit(handle, point, description) {
  const hit = await handle.evaluate((element, position) => {
    const observed = element.ownerDocument.elementFromPoint(position.x, position.y);
    return {
      matched: observed === element || element.contains(observed),
      observed: observed?.getAttribute('data-testid') ?? observed?.tagName ?? null,
      expected: element.getAttribute('data-testid')
    };
  }, point);
  assert.equal(
    hit.matched,
    true,
    `${description} must be under (${point.x}, ${point.y}); expected ${hit.expected}; observed ${hit.observed}`
  );
  return hit;
}

/** Callers finish scrolling before passing viewport coordinates; dragging never scrolls. */
export async function dragHandle(page, handle, destination, destinationHandle) {
  const bounds = await handle.boundingBox();
  assert.notEqual(bounds, null, 'Connection handle must have a visible bounding box');
  const origin = {x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2};
  const sourceHit = await checkHandleHit(handle, origin, 'Source handle');
  if (destinationHandle) await checkHandleHit(destinationHandle, destination, 'Input before drag');
  await page.mouse.move(origin.x, origin.y);
  await page.mouse.down();
  let destinationHit;
  try {
    await page.mouse.move(destination.x, destination.y, {steps: 24});
    // Auto-pan or layout during the gesture must fail instead of masquerading as a refusal.
    if (destinationHandle) {
      destinationHit = await checkHandleHit(destinationHandle, destination, 'Input at drop');
    }
  } finally {
    await page.mouse.up();
  }
  return {origin, destination, sourceHit, destinationHit};
}

export async function connectPorts(page, fromStepId, output, toStepId, input) {
  await frameGraph(page);
  const source = dock(page).getByTestId(TEST_IDS.outputPort(fromStepId, output));
  const destination = dock(page).getByTestId(TEST_IDS.inputPort(toStepId, input));
  await source.scrollIntoViewIfNeeded();
  await destination.scrollIntoViewIfNeeded();
  // Neither endpoint is measured until both scroll operations have completed.
  const bounds = await destination.boundingBox();
  assert.notEqual(bounds, null, `Input ${toStepId}.${input} must be visible before dragging`);
  return dragHandle(
    page,
    source,
    {
      x: bounds.x + bounds.width / 2,
      y: bounds.y + bounds.height / 2
    },
    destination
  );
}

export async function setParameter(page, name, value) {
  const control = dock(page).getByTestId(TEST_IDS.parameter(name));
  const kind = await control.evaluate(element => ({
    tag: element.tagName,
    type: element.type,
    role: element.getAttribute('role')
  }));
  if (kind.role === 'radiogroup') {
    await control.locator('input').evaluateAll((elements, selected) => {
      if (!elements.some(element => element.value === String(selected))) {
        throw new Error(
          `No radio value ${selected}; saw ${elements.map(element => element.value)}`
        );
      }
    }, value);
    await control.locator(`input[value="${String(value)}"]`).check();
  } else if (kind.tag === 'SELECT') await control.selectOption(String(value));
  else if (kind.type === 'checkbox') await control.setChecked(Boolean(value));
  else {
    await control.fill(String(value));
    await control.blur();
  }
}

export async function screenshot(page, directory, name) {
  await mkdir(directory, {recursive: true});
  await page.screenshot({path: join(directory, `${name}.png`), fullPage: true});
}

export async function useFirstTemplate(page) {
  await dock(page).getByTestId(TEST_IDS.templates).click();
  const gallery = page.getByRole('dialog', {name: 'Workflow templates', exact: true});
  const button = gallery.locator('[data-testid^="workflow-template-"]').first();
  const templateId = (await button.getAttribute('data-testid')).slice('workflow-template-'.length);
  await button.click();
  await gallery.waitFor({state: 'hidden'});
  return templateId;
}

export async function runAll(page) {
  const before = (await getWorkbenchState(page)).runLog[0]?.runId;
  await dock(page).getByTestId(TEST_IDS.runAll).click();
  await waitForWorkbench(
    page,
    'Run must finish and append a new run record',
    previous => {
      const state = window.__keplerWorkflowTest.store.getState();
      return !state.runner.running && state.runLog.length > 0 && state.runLog[0].runId !== previous;
    },
    before
  );
  const state = await getWorkbenchState(page);
  assert.equal(state.runLog[0].status, 'done', `Run failed: ${JSON.stringify(state)}`);
  for (const step of state.document.steps) await waitForStepStatus(page, step.id, 'done');
  return state;
}

/** Each executable shares the watchdog, diagnostics, screenshot path and guaranteed close. */
export async function runAcceptance(name, scenario) {
  const session = await launchWorkbench(process.argv[2]);
  const directory = process.argv[3] || './workflow-screenshots';
  try {
    const result = await scenario(session.page, directory);
    assert.deepEqual(session.failures, [], `Browser errors: ${JSON.stringify(session.failures)}`);
    console.log(JSON.stringify({ok: true, script: name, ...result}));
  } catch (error) {
    console.error(`${name}: ${error.stack ?? error}`);
    if (!session.page.isClosed()) {
      console.error(
        JSON.stringify({failures: session.failures, state: await getWorkbenchState(session.page)})
      );
      await screenshot(session.page, directory, `${name}-failure`);
    }
    process.exitCode = 1;
  } finally {
    clearTimeout(session.watchdog);
    await session.browser.close();
  }
}
