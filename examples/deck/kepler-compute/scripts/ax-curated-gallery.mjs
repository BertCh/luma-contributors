// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Usage: node scripts/ax-curated-gallery.mjs [host URL] [screenshot directory]
// External WebGPU/browser gate: each formerly disabled native sample reaches a GPU result
// from one Run sample click. Numerical recipe parity belongs to the dedicated curated specs.
import {
  dock,
  getWorkbenchState,
  openDock,
  runAcceptance,
  screenshot,
  TEST_IDS,
  waitForWorkbench
} from './workflow-utils.mjs';

const ANALYSES = [
  'drainage-network',
  'surface-preparation',
  'zonal-exposure',
  'regional-raster-change',
  'landscape-metrics',
  'dasymetric-redistribution',
  'cell-flow-topology',
  'drive-time-bands'
];

await runAcceptance('ax-curated-gallery', async (page, directory) => {
  await openDock(page);
  const hasDevice = await page.evaluate(() =>
    Boolean(window.__keplerWorkflowTest.store.host.getDevice())
  );
  if (!hasDevice) throw new Error('AX curated gallery requires a real WebGPU device.');
  const records = [];
  for (const analysisId of ANALYSES) {
    const gallery = page.getByTestId('ax-workflow-gallery');
    if (!(await gallery.isVisible())) await dock(page).getByTestId(TEST_IDS.templates).click();
    await gallery.waitFor();
    const button = gallery.getByTestId(`ax-sample-run-${analysisId}`);
    if (!(await button.isEnabled())) {
      throw new Error(`${analysisId}: Run sample is disabled: ${await button.getAttribute('title')}`);
    }
    const previousRunId = (await getWorkbenchState(page)).runLog[0]?.runId;
    await button.click();
    await waitForWorkbench(
      page,
      `${analysisId}: one sample action must append a finished run`,
      previous => {
        const state = window.__keplerWorkflowTest.store.getState();
        if (state.notice?.kind === 'error') throw new Error(state.notice.message);
        return Boolean(
          !state.runner?.running && state.runLog[0] && state.runLog[0].runId !== previous
        );
      },
      previousRunId,
      60000
    );
    const state = await getWorkbenchState(page);
    const analysis = state.steps['analysis'];
    if (state.runLog[0].status !== 'done' || analysis?.result?.engine !== 'gpu') {
      throw new Error(`${analysisId}: expected GPU result: ${JSON.stringify(state)}`);
    }
    const outputs = Object.values(analysis.result.outputs);
    if (!outputs.some(output => output.rowCount > 0)) {
      throw new Error(`${analysisId}: no nonempty output: ${JSON.stringify(outputs)}`);
    }
    records.push({analysisId, actions: 1, engine: analysis.result.engine, outputs});
    await screenshot(page, directory, `ax-curated-${analysisId}`);
  }
  return {analyses: records};
});
