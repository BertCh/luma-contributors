// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {
  dock,
  frameGraph,
  openDock,
  runAcceptance,
  runAll,
  screenshot,
  TEST_IDS
} from './workflow-utils.mjs';

await runAcceptance('workflow-templates', async (page, directory) => {
  await openDock(page);
  await dock(page).getByTestId(TEST_IDS.templates).click();
  const gallery = page.getByRole('dialog', {name: 'Workflow templates', exact: true});
  const templateIds = await gallery
    .locator('[data-testid^="workflow-template-"]')
    .evaluateAll(elements =>
      elements.map(element => element.dataset.testid.slice('workflow-template-'.length))
    );
  assert.equal(templateIds.length > 0, true, 'Templates dialog must contain at least one template');
  assert.equal(new Set(templateIds).size, templateIds.length, 'Template ids must be unique');
  const records = [];
  for (const templateId of templateIds) {
    if (!(await gallery.isVisible())) await dock(page).getByTestId(TEST_IDS.templates).click();
    await gallery.getByTestId(TEST_IDS.template(templateId)).click();
    await gallery.waitFor({state: 'hidden'});
    const state = await runAll(page);
    const run = state.runLog[0];
    assert.equal(
      run.steps.length,
      state.document.steps.length,
      `${templateId}: run record must contain every template step`
    );
    for (const step of run.steps) {
      assert.equal(step.status, 'done', `${templateId}/${step.stepId}: ${step.error?.message}`);
      assert.equal(step.cacheHit, false, `${templateId}/${step.stepId} must execute its first run`);
      assert.equal(
        Number.isFinite(step.durationMilliseconds) && step.durationMilliseconds >= 0,
        true,
        `${templateId}/${step.stepId}: duration must be finite and nonnegative`
      );
      assert.equal(
        ['host', 'gpu', 'cpu-reference'].includes(step.engine),
        true,
        `${templateId}/${step.stepId}: expected an engine; saw ${step.engine}`
      );
    }
    records.push({
      templateId,
      durationMilliseconds: run.durationMilliseconds,
      steps: run.steps.map(step => ({
        stepId: step.stepId,
        engine: step.engine,
        durationMilliseconds: step.durationMilliseconds,
        fallbackReason: step.fallbackReason
      }))
    });
    await frameGraph(page);
    await screenshot(page, directory, `workflow-templates-${templateId}-finished`);
  }
  const hasDevice = await page.evaluate(() =>
    Boolean(window.__keplerWorkflowTest.store.host.getDevice())
  );
  assert.equal(
    !hasDevice || records.some(record => record.steps.some(step => step.engine === 'gpu')),
    true,
    `GPU device is present, but no template used GPU: ${JSON.stringify(records)}`
  );
  return {hasDevice, templates: records};
});
