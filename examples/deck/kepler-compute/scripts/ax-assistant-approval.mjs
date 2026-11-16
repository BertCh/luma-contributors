// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Deterministic host/tool + approval UI qualification; this does not call a language model.
// node scripts/ax-assistant-approval.mjs <url> <evidence-directory>
import {runAcceptance, screenshot, waitForWorkbench} from './workflow-utils.mjs';

const proposal = {
  title: 'Assistant approval qualification',
  steps: [
    {
      id: 'observations',
      operationId: 'source-geojson',
      parameters: {
        coordinateSystem: 'EPSG:4326',
        profileDeclarations: JSON.stringify({version: 1, rowSemantics: 'events'}),
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
    },
    {
      id: 'density',
      operationId: 'kernel-density-grid',
      inputs: {points: 'observations.data'},
      parameters: {cellSize: 500, bandwidth: 750}
    }
  ]
};

async function callAssistant(page, call, approved = false) {
  return page.evaluate(async ({call, approved}) => {
    const {runWorkbenchAssistantTool} = await import('/src/workflow/workbench-assistant.ts');
    return runWorkbenchAssistantTool(window.__keplerWorkflowTest.store, call, {approved});
  }, {call, approved});
}

async function readCounts(page) {
  return page.evaluate(() => {
    const state = window.__keplerWorkflowTest.store.getState();
    return {workflows: state.workflows.length, runs: state.runLog.length, draft: Boolean(state.draft)};
  });
}

await runAcceptance('ax-assistant-approval', async (page, directory) => {
  await page.evaluate(async () => { await window.__keplerWorkflowTest.store.clear(); });
  const discovery = await callAssistant(page, {name: 'listAnalysisTemplates', arguments: {query: 'density'}});
  if (!discovery.templates.some(template => template.id === 'density-surface'))
    throw new Error(`AX assistant discovery failed: ${JSON.stringify(discovery)}`);
  const refused = await callAssistant(page, {
    name: 'defineWorkflow',
    arguments: {templateId: 'exact-earthquake-prediction', datasets: {}, columns: {}}
  }, true);
  if (refused.status !== 'error' || !/not available/.test(refused.message))
    throw new Error(`AX assistant refusal failed: ${JSON.stringify(refused)}`);

  const draft = await callAssistant(page, {name: 'defineWorkflow', arguments: proposal});
  if (draft.status !== 'awaiting-approval')
    throw new Error(`AX assistant proposal failed: ${JSON.stringify(draft)}`);
  const review = page.getByRole('region', {name: 'Proposed workflow', exact: true});
  await review.waitFor({state: 'visible'});
  if (!(await review.innerText()).includes('750'))
    throw new Error('Whole-plan review must show the proposed bandwidth.');
  let counts = await readCounts(page);
  if (counts.workflows || counts.runs || !counts.draft)
    throw new Error(`A proposal must only create a draft: ${JSON.stringify(counts)}`);
  await screenshot(page, directory, 'ax-assistant-whole-plan-review');
  // Simulate a source declaration changing while the frozen analysis settings remain under
  // review. The first acceptance must keep the draft visible and show the fresh contradiction.
  await page.evaluate(() => {
    const store = window.__keplerWorkflowTest.store;
    const draft = structuredClone(store.getState().draft);
    draft.document.steps.find(step => step.id === 'observations').parameters.coordinateSystem.value = 'planar';
    store.proposeDraft(draft);
  });
  await page.getByTestId('workflow-draft-accept').click();
  await waitForWorkbench(page, 'Changed validation warnings must remain in the review draft', () => {
    const state = window.__keplerWorkflowTest.store.getState();
    return Boolean(state.draft?.issues.some(issue => issue.severity === 'warning' && /planar/.test(issue.message))) && state.workflows.length === 0;
  });
  await review.waitFor({state: 'visible'});
  if (!/planar/.test(await review.innerText()))
    throw new Error('Updated coordinate warnings must be visible before renewed whole-plan approval.');
  counts = await readCounts(page);
  if (counts.workflows || counts.runs || !counts.draft)
    throw new Error(`Changed warnings must prevent acceptance and execution: ${JSON.stringify(counts)}`);
  await screenshot(page, directory, 'ax-assistant-updated-warnings');
  await page.getByTestId('workflow-draft-reject').click();
  await review.waitFor({state: 'hidden'});
  counts = await readCounts(page);
  if (counts.workflows || counts.runs || counts.draft)
    throw new Error(`Discard must leave no workflow or run: ${JSON.stringify(counts)}`);

  const secondDraft = await callAssistant(page, {name: 'defineWorkflow', arguments: proposal});
  if (secondDraft.status !== 'awaiting-approval')
    throw new Error(`Second proposal failed: ${JSON.stringify(secondDraft)}`);
  await review.waitFor({state: 'visible'});
  await page.getByTestId('workflow-draft-accept').click();
  await review.waitFor({state: 'hidden'});
  counts = await readCounts(page);
  if (counts.workflows !== 1 || counts.runs || counts.draft)
    throw new Error(`Approval must add the whole plan without running: ${JSON.stringify(counts)}`);
  const workflowId = await page.evaluate(() => window.__keplerWorkflowTest.store.getState().activeWorkflowId);
  const run = {name: 'runWorkflow', arguments: {workflowId}};
  const unapproved = await callAssistant(page, run);
  if (unapproved.status !== 'needs-approval' || (await readCounts(page)).runs)
    throw new Error(`An unapproved run must stay refused: ${JSON.stringify(unapproved)}`);
  const approved = await callAssistant(page, run, true);
  if (approved.status !== 'started')
    throw new Error(`Approved run failed to start: ${JSON.stringify(approved)}`);
  await waitForWorkbench(page, 'The approved whole plan must complete', () => {
    const state = window.__keplerWorkflowTest.store.getState();
    return !state.runner.running && state.runLog.length === 1;
  });
  const record = await page.evaluate(() => window.__keplerWorkflowTest.store.getState().runLog[0]);
  if (record.status !== 'done' || record.steps.length !== 2)
    throw new Error(`Approved plan execution failed: ${JSON.stringify(record)}`);
  await screenshot(page, directory, 'ax-assistant-approved-result');
  return {checks: ['template discovery', 'unsupported refusal', 'whole-plan review', 'updated warnings retained for renewed review', 'discard', 'accept without execution', 'run approval', 'successful run']};
});
