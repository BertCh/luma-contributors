// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {
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
  useFirstTemplate,
  waitForRunIdle,
  waitForStepStatus,
  waitForWorkbench
} from './workflow-utils.mjs';

export function checkMinimumPointsProblem(message) {
  assert.equal(
    message,
    'Find clusters: Minimum points (including the point) must be between 1 and 1000; got 0.',
    'Invalid Minimum points must name the step, parameter, exact range and rejected value'
  );
}

export function checkInvalidDraft(before, after, stepId, parameter) {
  const previous = before.document.steps.find(step => step.id === stepId);
  const current = after.document.steps.find(step => step.id === stepId);
  assert.deepEqual(
    current.parameters[parameter],
    previous.parameters[parameter],
    `${stepId}: an invalid ${parameter} draft must preserve the last valid binding`
  );
  assert.equal(
    after.steps[stepId].status,
    before.steps[stepId].status,
    `${stepId}: an invalid draft must preserve the step status`
  );
}

export function checkRunToHere(before, after, targetId) {
  const included = new Set();
  function includeStep(stepId) {
    if (included.has(stepId)) return;
    included.add(stepId);
    const step = before.document.steps.find(candidate => candidate.id === stepId);
    assert.notEqual(step, undefined, `Run to here: step ${stepId} must exist`);
    for (const connection of Object.values(step.inputs)) includeStep(connection.stepId);
  }
  includeStep(targetId);
  const record = after.runLog[0];
  assert.equal(record.status, 'done', `Run to here failed: ${JSON.stringify(record)}`);
  assert.equal(record.target, targetId, 'Run to here must target the selected middle step');
  const expected = [...included].filter(stepId => before.steps[stepId].status !== 'done');
  assert.deepEqual(
    record.steps.map(step => step.stepId).sort(),
    expected.sort(),
    'Run to here must list exactly the target and upstream steps that needed a run'
  );
  // A current upstream result is read directly; it is not a cache-hit entry in this run.
  for (const stepId of included) {
    if (before.steps[stepId].status !== 'done') continue;
    const previous = before.steps[stepId].result;
    const current = after.steps[stepId];
    assert.equal(current.status, 'done', `${stepId}: current upstream must stay done`);
    assert.equal(
      Number.isFinite(previous?.completedAt),
      true,
      `${stepId}: the source result must expose a finite completedAt`
    );
    assert.equal(
      current.result?.completedAt,
      previous.completedAt,
      `${stepId}: Run to here must preserve the upstream result completedAt`
    );
  }
}

async function runEditScenario(page, directory) {
  await openDock(page);
  const templateId = await useFirstTemplate(page);
  const baseline = await runAll(page);
  assert.equal(templateId, 'cluster-footprints', 'First template must be Cluster footprints');
  await frameGraph(page);
  await screenshot(page, directory, 'workflow-edit-initial-done');
  await dock(page).getByTestId(TEST_IDS.node('clusters')).click();
  await setParameter(page, 'epsilon', 100);
  await waitForStepStatus(page, 'samples', 'done');
  for (const stepId of ['clusters', 'members', 'footprints']) {
    await waitForStepStatus(page, stepId, 'stale');
  }
  await screenshot(page, directory, 'workflow-edit-stale');
  const beforeTargetedRun = await getWorkbenchState(page);
  assert.equal(
    beforeTargetedRun.steps.samples.result.completedAt,
    baseline.steps.samples.result.completedAt,
    'Editing the middle step must preserve the source result completedAt'
  );
  const previousRunId = baseline.runLog[0].runId;
  await dock(page).getByTestId(TEST_IDS.runToHere).click();
  await waitForWorkbench(
    page,
    'Run to here must append a completed run record',
    previous => {
      const state = window.__keplerWorkflowTest.store.getState();
      return !state.runner.running && state.runLog[0]?.runId !== previous;
    },
    previousRunId
  );
  let state = await getWorkbenchState(page);
  const targeted = state.runLog[0];
  checkRunToHere(beforeTargetedRun, state, 'clusters');
  assert.deepEqual(
    targeted.steps.map(step => [step.stepId, step.cacheHit]),
    [['clusters', false]],
    'Run to here must execute only the newly stale middle step; current sources are omitted'
  );
  await waitForStepStatus(page, 'members', 'stale');
  await waitForStepStatus(page, 'footprints', 'stale');
  await screenshot(page, directory, 'workflow-edit-run-to-here');
  const runCount = state.runLog.length;
  await dock(page).getByTestId(TEST_IDS.undo).click();
  for (const step of baseline.document.steps) await waitForStepStatus(page, step.id, 'done');
  state = await getWorkbenchState(page);
  assert.deepEqual(state.document, baseline.document, 'Undo must restore the original document');
  assert.equal(state.runLog.length, runCount, 'Undo must recover cached results without a run');
  await screenshot(page, directory, 'workflow-edit-undo-done');
  await dock(page).getByTestId(TEST_IDS.redo).click();
  await waitForStepStatus(page, 'members', 'stale');
  assert.equal(
    (await getWorkbenchState(page)).runLog.length,
    runCount,
    'Redo must not start a run when auto-run is off'
  );
  await screenshot(page, directory, 'workflow-edit-redo');
  await dock(page).getByTestId(TEST_IDS.nodeDisable('clusters')).click();
  await waitForStepStatus(page, 'clusters', 'disabled');
  await waitForStepStatus(page, 'members', 'blocked');
  await waitForStepStatus(page, 'footprints', 'blocked');
  await screenshot(page, directory, 'workflow-edit-disabled');
  const beforeDelete = (await getWorkbenchState(page)).document;
  await dock(page).getByTestId(TEST_IDS.node('members')).click();
  await dock(page).getByTestId(TEST_IDS.canvas).focus();
  await page.keyboard.press('Delete');
  state = await getWorkbenchState(page);
  assert.equal(
    state.document.steps.some(step => step.id === 'members'),
    false,
    'Delete must remove the selected middle step'
  );
  assert.equal(
    Object.values(state.document.steps.find(step => step.id === 'footprints').inputs).some(
      connection => connection.stepId === 'members'
    ),
    false,
    'Delete must remove dangling edges'
  );
  await screenshot(page, directory, 'workflow-edit-deleted');
  await dock(page).getByTestId(TEST_IDS.undo).click();
  assert.deepEqual(
    (await getWorkbenchState(page)).document,
    beforeDelete,
    'Undo delete must restore the exact step, parameters and edges'
  );
  await screenshot(page, directory, 'workflow-edit-undo-delete');
  await dock(page).getByTestId(TEST_IDS.nodeDisable('clusters')).click();
  await runAll(page);

  // An independent polygon branch makes this a kind refusal, without also introducing a cycle.
  await page.evaluate(() =>
    window.__keplerWorkflowTest.store.edit('refusal-fixture', document => ({
      ...document,
      steps: [
        ...document.steps,
        {
          id: 'refusal-polygons',
          operationId: 'geometry-point-buffer',
          inputs: {points: {stepId: 'samples', output: 'data'}},
          parameters: {},
          position: {x: 300, y: 260}
        },
        {
          id: 'refusal-points',
          operationId: 'random-points-in-polygon',
          inputs: {polygons: {stepId: 'refusal-polygons', output: 'buffers'}},
          parameters: {pointsPerPolygon: {from: 'static', value: 1}},
          position: {x: 600, y: 260}
        }
      ]
    }))
  );
  const beforeConnection = (await getWorkbenchState(page)).document;
  await connectPorts(page, 'refusal-points', 'random-points', 'clusters', 'points');
  assert.deepEqual(
    (await getWorkbenchState(page)).document.steps.find(step => step.id === 'clusters').inputs
      .points,
    {stepId: 'refusal-points', output: 'random-points'},
    'Positive control: a compatible drag must replace the connection on the same points input'
  );
  await dock(page).getByTestId(TEST_IDS.undo).click();
  assert.deepEqual(
    (await getWorkbenchState(page)).document,
    beforeConnection,
    'Undo must restore the original edge before testing the polygon refusal'
  );
  const refusalGesture = await connectPorts(
    page,
    'refusal-polygons',
    'buffers',
    'clusters',
    'points'
  );
  assert.equal(
    refusalGesture.destinationHit.expected,
    TEST_IDS.inputPort('clusters', 'points'),
    'Refusal must release over the points-only input, verified by a DOM hit test'
  );
  assert.equal(
    refusalGesture.destinationHit.matched,
    true,
    `Refusal drag must reach the intended input; observed ${JSON.stringify(refusalGesture)}`
  );
  assert.deepEqual(
    (await getWorkbenchState(page)).document,
    beforeConnection,
    'Dragging a polygon output to a points-only input must leave the graph unchanged'
  );
  await screenshot(page, directory, 'workflow-edit-refused-connection');
  await dock(page).getByTestId(TEST_IDS.node('clusters')).click();
  const beforeInvalid = await getWorkbenchState(page);
  await setParameter(page, 'minimumPoints', 0);
  const problem = dock(page).getByTestId(TEST_IDS.parameterError('minimumPoints'));
  await problem.waitFor();
  assert.equal(await problem.isVisible(), true, 'Invalid parameter problem must be visible');
  assert.equal(
    await problem.getAttribute('role'),
    'alert',
    'Invalid parameter problem must be an alert'
  );
  checkMinimumPointsProblem(await problem.innerText());
  assert.equal(
    await dock(page).getByTestId(TEST_IDS.parameter('minimumPoints')).getAttribute('aria-invalid'),
    'true',
    'Invalid draft must mark the control aria-invalid'
  );
  checkInvalidDraft(beforeInvalid, await getWorkbenchState(page), 'clusters', 'minimumPoints');
  await screenshot(page, directory, 'workflow-edit-invalid-parameter');

  // Changing selection discards the local draft without committing a new parameter binding.
  await dock(page).getByTestId(TEST_IDS.node('samples')).click();
  await dock(page).getByTestId(TEST_IDS.node('clusters')).click();
  const beforeDisconnect = await getWorkbenchState(page);
  await dock(page)
    .getByTestId(TEST_IDS.inputBinding('points'))
    .selectOption({label: 'Not connected'});
  await waitForStepStatus(page, 'clusters', 'not-configured');
  await waitForStepStatus(page, 'members', 'blocked');
  await waitForStepStatus(page, 'footprints', 'blocked');
  assert.equal(
    (await getWorkbenchState(page)).steps.clusters.issues.some(
      issue => issue.code === 'input-required' && issue.port === 'points'
    ),
    true,
    'Disconnecting the required Points input must produce an input-required issue'
  );
  await screenshot(page, directory, 'workflow-edit-required-input-disconnected');
  await dock(page).getByTestId(TEST_IDS.undo).click();
  assert.deepEqual(
    (await getWorkbenchState(page)).document,
    beforeDisconnect.document,
    'Undo must restore the exact document and required input connection'
  );
  for (const stepId of ['clusters', 'members', 'footprints']) {
    await waitForStepStatus(page, stepId, beforeDisconnect.steps[stepId].status);
  }
  await screenshot(page, directory, 'workflow-edit-undo-required-input');
  await waitForRunIdle(page);
  return {
    templateId,
    targetedSteps: targeted.steps.map(step => ({stepId: step.stepId, cacheHit: step.cacheHit})),
    runCount
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await runAcceptance('workflow-edit', runEditScenario);
}
