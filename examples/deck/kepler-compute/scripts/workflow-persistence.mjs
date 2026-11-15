// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {
  getWorkbenchState,
  openDock,
  POLYGON_TEXT,
  runAcceptance,
  runAll,
  screenshot,
  waitForWorkbench
} from './workflow-utils.mjs';

/** The injected parser lets Node regressions exercise the same preflight without a browser. */
export async function restoreWorkflowCollection(value, parseCollection) {
  const parse =
    parseCollection ?? (await import('/@id/@luma-contributors/kepler')).parseWorkflowCollection;
  let collection;
  try {
    collection = parse(value);
  } catch (error) {
    return {
      restored: false,
      code: typeof error?.code === 'string' ? error.code : null,
      message: error instanceof Error ? error.message : String(error)
    };
  }
  window.__keplerWorkflowTest.store.restore(collection);
  return {restored: true};
}

async function runPersistenceScenario(page, directory) {
  await openDock(page);
  await page.evaluate(text => {
    const store = window.__keplerWorkflowTest.store;
    store.addWorkflow({
      schemaVersion: 1,
      id: 'persistence-acceptance',
      title: 'Persistence acceptance',
      inputs: [],
      settings: {engine: 'auto', autoRun: false},
      steps: [
        {
          id: 'source',
          operationId: 'source-geojson',
          inputs: {},
          parameters: {text: {from: 'static', value: text}},
          position: {x: 0, y: 0}
        },
        {
          id: 'points',
          operationId: 'random-points-in-polygon',
          inputs: {polygons: {stepId: 'source', output: 'data'}},
          parameters: {pointsPerPolygon: {from: 'static', value: 12}},
          position: {x: 300, y: 0}
        }
      ]
    });
  }, POLYGON_TEXT);
  await runAll(page);
  const saved = await page.evaluate(() => window.__keplerWorkflowTest.store.serialize());
  assert.notEqual(saved, undefined, 'A populated workbench must serialize a workflow collection');
  await screenshot(page, directory, 'workflow-persistence-before-save');
  // dev-test-hooks.ts exposes no sandbox round-trip or host-extension hook. Exercise the
  // workflow collection seam directly; this does not claim sandbox-host integration coverage.
  await screenshot(page, directory, 'workflow-persistence-saved');
  await page.evaluate(async () => window.__keplerWorkflowTest.store.clear());
  const cleared = await getWorkbenchState(page);
  assert.equal(cleared.workflows.length, 0, 'Clear must remove all workflows before replay');
  assert.equal(cleared.activeWorkflowId, null, 'Clear must remove the active workflow');
  assert.deepEqual(cleared.steps, {}, 'Clear must discard the previous runner results');
  await screenshot(page, directory, 'workflow-persistence-cleared');
  const restored = await page.evaluate(restoreWorkflowCollection, saved);
  assert.deepEqual(
    restored,
    {restored: true},
    `Validated workflow collection must restore successfully; observed ${JSON.stringify(restored)}`
  );
  await waitForWorkbench(
    page,
    'Collection restore must restore both workflow steps as ready',
    () => {
      const state = window.__keplerWorkflowTest.store.getState();
      return (
        state.document?.steps.length === 2 &&
        state.document.steps.every(step => state.runner.steps[step.id].status === 'ready')
      );
    }
  );
  assert.deepEqual(
    await page.evaluate(() => window.__keplerWorkflowTest.store.serialize()),
    saved,
    'Collection restore must reproduce store.serialize() exactly, without cached results'
  );
  await screenshot(page, directory, 'workflow-persistence-replayed-ready');

  const beforeMalformed = await getWorkbenchState(page);
  const malformed = {...saved, workflows: 'not-an-array'};
  const refusal = await page.evaluate(restoreWorkflowCollection, malformed);
  assert.deepEqual(
    refusal,
    {restored: false, code: 'malformed', message: 'collection.workflows: must be an array.'},
    'Malformed collection must throw during parsing before store.restore is called'
  );
  const afterMalformed = await getWorkbenchState(page);
  assert.deepEqual(
    afterMalformed,
    beforeMalformed,
    'Malformed collection parsing must leave the current workbench unchanged'
  );
  assert.deepEqual(
    await page.evaluate(() => window.__keplerWorkflowTest.store.serialize()),
    saved,
    'Malformed collection parsing must preserve the full saved workbench'
  );
  await screenshot(page, directory, 'workflow-persistence-malformed-refused');
  return {
    route: 'workflow-collection',
    workflowId: saved.activeWorkflowId,
    savedCharacters: JSON.stringify(saved).length,
    restoredStatuses: Object.fromEntries(
      Object.entries(afterMalformed.steps).map(([stepId, step]) => [stepId, step.status])
    ),
    refusal
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await runAcceptance('workflow-persistence', runPersistenceScenario);
}
