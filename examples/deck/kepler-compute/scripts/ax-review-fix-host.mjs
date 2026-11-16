// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Lead-run WebGPU/browser regression: decoded loaded raster -> Use my data -> real drainage.
// Also checks transient invalid settings and warning-free edits after assistant run review.
// Usage: node scripts/ax-review-fix-host.mjs [host URL] [evidence directory]
import {
  dock,
  getWorkbenchState,
  openDock,
  runAcceptance,
  screenshot,
  TEST_IDS,
  waitForWorkbench
} from './workflow-utils.mjs';
import {injectElevationRaster} from './raster-panels-utils.mjs';

await runAcceptance('ax-review-fix-host', async (page, directory) => {
  await openDock(page);
  await injectElevationRaster(page, 'review-loaded-elevation', 'Review loaded elevation');
  const gallery = page.getByTestId('ax-workflow-gallery');
  if (!(await gallery.isVisible())) await dock(page).getByTestId(TEST_IDS.templates).click();
  await gallery.getByTestId('ax-use-data-drainage-network').click();
  const mapping = page.getByTestId('ax-analysis-mapping');
  await mapping.waitFor();
  const dataset = mapping.getByTestId('ax-mapping-dataset-elevation');
  const selectedId = await page.evaluate(() => {
    const host = window.__keplerWorkflowTest.store.host;
    return host
      .listDatasets()
      .find(dataset => dataset.label.startsWith('Review loaded elevation · tile '))?.id;
  });
  if (!selectedId)
    throw new Error('Loaded decoded raster is missing from production host discovery.');
  await dataset.selectOption(selectedId);
  await waitForWorkbench(page, 'Loaded raster mapping becomes runnable', () => {
    const run = document.querySelector('[data-testid="ax-mapping-run"]');
    return run && !run.disabled;
  });
  const before = await getWorkbenchState(page);
  const threshold = mapping.getByTestId('ax-mapping-parameter-streamThreshold');
  const original = await threshold.inputValue();
  await threshold.fill('-1');
  await threshold.blur();
  const run = mapping.getByTestId('ax-mapping-run');
  if (!(await run.isDisabled())) throw new Error('Invalid raster threshold left Run enabled.');
  if ((await getWorkbenchState(page)).runLog.length !== before.runLog.length)
    throw new Error('Invalid parameter text executed a workflow.');
  await screenshot(page, directory, 'invalid-mapping-parameter');
  await threshold.fill(original);
  await threshold.blur();
  if (!(await run.isEnabled())) throw new Error('Correcting parameter text did not re-enable Run.');
  await run.click();
  await waitForWorkbench(
    page,
    'Loaded raster produces a successful GPU drainage result',
    previousRun => {
      const state = window.__keplerWorkflowTest.store.getState();
      if (state.notice?.kind === 'error') throw new Error(state.notice.message);
      return (
        !state.runner?.running &&
        state.runLog[0]?.status === 'done' &&
        state.runLog[0].runId !== previousRun
      );
    },
    before.runLog[0]?.runId,
    60000
  );
  const result = await page.evaluate(() => {
    const store = window.__keplerWorkflowTest.store;
    const state = store.getState();
    const source = state.runner.steps['source-elevation'].result.outputs.data;
    const analysis = state.runner.steps.analysis.result;
    return {
      kind: source.kind,
      datasetId: source.table.id,
      cells: source.raster.grid.width * source.raster.grid.height,
      band: source.raster.band.index,
      invalid: Array.from(source.raster.band.validity).filter(value => !value).length,
      engine: analysis.engine,
      outputNames: Object.keys(analysis.outputs)
    };
  });
  if (
    result.kind !== 'raster' ||
    result.datasetId !== selectedId ||
    result.band !== 1 ||
    result.cells !== 1024 ||
    !result.invalid ||
    result.engine !== 'gpu'
  )
    throw new Error(`Loaded-raster handoff mismatch: ${JSON.stringify(result)}`);
  await screenshot(page, directory, 'loaded-raster-drainage-result');
  const approval = await page.evaluate(async () => {
    const {runWorkbenchAssistantTool} = await import('/src/workflow/workbench-assistant.ts');
    const store = window.__keplerWorkflowTest.store;
    const workflowId = store.getState().activeWorkflowId;
    const call = {name: 'runWorkflow', arguments: {workflowId}};
    const reviewed = runWorkbenchAssistantTool(store, call);
    const runs = store.getState().runLog.length;
    store.edit('review-parameter-change', document => ({
      ...document,
      steps: document.steps.map(step =>
        step.id === 'analysis'
          ? {
              ...step,
              parameters: {...step.parameters, streamThreshold: {from: 'static', value: 10}}
            }
          : step
      )
    }));
    const approvedOldCall = runWorkbenchAssistantTool(store, call, {approved: true});
    const newReview = runWorkbenchAssistantTool(store, call);
    const changedTarget = runWorkbenchAssistantTool(
      store,
      {...call, arguments: {workflowId, target: 'source-elevation'}},
      {approved: true}
    );
    return {
      reviewed,
      approvedOldCall,
      newReview,
      changedTarget,
      runs,
      after: store.getState().runLog.length
    };
  });
  if (
    approval.approvedOldCall.status !== 'needs-approval' ||
    approval.changedTarget.status !== 'needs-approval' ||
    approval.after !== approval.runs
  )
    throw new Error(
      `Assistant approval survived a changed plan or target: ${JSON.stringify(approval)}`
    );
  return {raster: result, approval};
});
