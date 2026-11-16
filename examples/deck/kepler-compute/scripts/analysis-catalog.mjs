// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check for the contributor catalog panel: discovery with reasons, a typed run of the
// built-in network service area on the GPU contributors (engine shown in the panel), the CPU
// reference engine as an explicit choice, GPU-only overlay and noding operations, serialization
// restore, and staging the result into the Sources workflow, then publishing it as a native dataset.
// Usage: node scripts/analysis-catalog.mjs [URL] (default: http://127.0.0.1:5318/?rows=256).
import {setAnalysisSheetParameter} from './analysis-sheet-controls.mjs';
import {exerciseValidationCatalog} from './analysis-catalog-validation.mjs';
import {checkPointCloudCatalogWorkflow} from './analysis-catalog-pointcloud.mjs';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {
  NETWORK_CATEGORICAL_OPERATIONS,
  runNetworkCategoricalCatalog
} from './network-categorical-catalog-utils.mjs';
import {exerciseTrajectoryCatalog} from './catalog-trajectory.mjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (
    message.type() === 'error' &&
    /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())
  )
    failures.push(message.text());
});
const watchdog = setTimeout(() => {
  console.error('Catalog workflow exceeded 360 seconds');
  void browser.close();
}, 360000);

/** Stage geometry through the real preparation UI, then require a published polygon layer. */
async function publishPolygonResult(panel, label, expectedRowCount) {
  const existingIds = await page.evaluate(() =>
    window.__keplerComputeTest.getState().publications.map(source => source.id)
  );
  await panel.getByTestId('catalog-publish').click();
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  const preparation = workspace.getByRole('region', {name: `Prepare ${label}`, exact: true});
  await preparation.waitFor({timeout: 15000});
  // Staging must carry the binding's geometry role; choosing it here would hide a lost role.
  assert.equal(
    await workspace.getByLabel('Source geometry column', {exact: true}).inputValue(),
    'geometry'
  );
  await preparation.getByLabel('Float64 conversion', {exact: true}).selectOption('exact');
  await preparation.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await preparation.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    ({knownIds, rowCount}) => {
      const publication = window.__keplerComputeTest
        .getState()
        .publications.find(source => !knownIds.includes(source.id));
      if (publication?.error) throw new Error(publication.error);
      return publication?.rowCount === rowCount && publication.selectedCount === rowCount;
    },
    {knownIds: existingIds, rowCount: expectedRowCount},
    {timeout: 45000}
  );
  const publication = await page.evaluate(
    knownIds =>
      window.__keplerComputeTest
        .getState()
        .publications.find(source => !knownIds.includes(source.id)),
    existingIds
  );
  assert(publication.geometry?.sourceUploadBytes > 0, `${label} uploads resident geometry`);
  // Publication registers native data through addDataToMap, whose reducer closes the modal.
  // Wait for that transaction before reopening it, so a late registration cannot close it again.
  await page.waitForFunction(
    datasetId =>
      window.__keplerComputeTest
        .getState()
        .nativeLayers.some(
          candidate => candidate.dataId === datasetId && candidate.type === 'geojson'
        ),
    publication.id,
    {timeout: 15000}
  );
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const layer = configuration.config.visState.layers.find(
    candidate => candidate.config.dataId === publication.id
  );
  assert(layer, `${label} has a native layer`);
  assert.equal(layer.type, 'geojson');
  assert.equal(layer.config.columns.geojson, 'geometry');
  assert.equal(layer.config.label, label);
  // Resident datasets and its Remove controls live only inside the native preparation dialog.
  await openNativeData(page);
  await workspace.waitFor();
  await workspace.getByRole('button', {name: `Remove ${label}`, exact: true}).waitFor();
  await closeNativeDialog(page);
  // GeoJSON may contain points or lines: require the actual filled polygon primitive for this dataset.
  await page.waitForFunction(
    layerId =>
      window.__keplerComputeTest
        .renderedLayers()
        .some(
          candidate =>
            candidate.id === layerId &&
            candidate.layerClass === 'GPUSolidPolygonLayer' &&
            candidate.loaded &&
            candidate.visible &&
            candidate.indexBufferIds?.length > 0
        ),
    layer.id,
    {timeout: 45000}
  );
}

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  // The first load against a freshly started (or --force re-optimized) Vite server pre-bundles
  // dependencies and can take more than a minute; a warm load takes seconds.
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 240000});
  const panel = page.getByRole('region', {name: 'Analysis catalog', exact: true});
  await panel.waitFor();

  // Discovery: some contributors are runnable, the rest list a reason.
  const summary = await panel.getByTestId('catalog-summary').innerText();
  assert.match(summary, /contributors are runnable here/);
  await panel.getByRole('button', {name: /not runnable/}).click();
  assert.match(
    await panel.getByTestId('catalog-blocked').innerText(),
    /No executable host binding/
  );

  // Every new wrapper runs from its real built-in sample, with GPU dispatch visible in the UI.
  for (const [operationId, contributor] of NETWORK_CATEGORICAL_OPERATIONS) {
    await panel.getByLabel('Operation', {exact: true}).selectOption(operationId);
    await runNetworkCategoricalCatalog(page, panel, contributor);
  }
  // HEALPix cells stage as polygon geometry rather than a table of ids alone.
  await panel.getByLabel('Operation', {exact: true}).selectOption('healpix-cell-aggregation');
  await runNetworkCategoricalCatalog(page, panel, 'GPUHealpixCellAggregation');
  // The sample aggregates to one cell; polygon geometry is intentionally hidden in Result preview.
  assert.match(
    await panel.getByTestId('catalog-result').innerText(),
    /HEALPix cell aggregation: 1 rows/
  );
  await publishPolygonResult(panel, 'HEALPix cell aggregation', 1);

  await exerciseValidationCatalog(page, panel);

  // Service area on the sample network: statuses and source edge IDs.
  await panel.getByLabel('Operation', {exact: true}).selectOption('network-service-area');
  await setAnalysisSheetParameter(panel, 'coordinateSystem', 'planar');
  await setAnalysisSheetParameter(panel, 'onewayProperty', 'oneway');
  await setAnalysisSheetParameter(panel, 'maximumSnapDistance', 1);
  await setAnalysisSheetParameter(panel, 'budgets', [15]);
  await panel.getByTestId('catalog-check-inputs').click();
  await panel.getByTestId('catalog-run').click();
  const result = panel.getByTestId('catalog-result');
  await result.waitFor();
  const text = await result.innerText();
  assert.match(text, /reachedEdgeCount: 3/);
  assert.match(text, /unreachable-directed/);
  assert.match(text, /unreachable-disconnected/);

  // The panel names the engine that ran: GPU contributors with a device, CPU reference on request.
  assert.match(await panel.getByTestId('catalog-device').innerText(), /GPU device/);
  assert.match(
    await panel.getByTestId('catalog-engine').innerText(),
    /Engine: GPU \(.*GPUNetworkReachability/
  );
  await panel.getByLabel('Engine', {exact: true}).selectOption('cpu');
  await panel.getByTestId('catalog-check-inputs').click();
  await panel.getByTestId('catalog-run').click();
  await page.waitForFunction(() =>
    /Engine: CPU reference/.test(
      document.querySelector('[data-testid="catalog-engine"]')?.textContent ?? ''
    )
  );
  assert.match(await result.innerText(), /reachedEdgeCount: 3/);
  await panel.getByLabel('Engine', {exact: true}).selectOption('auto');

  // GPU-only operations: overlay intersection and crossing noding on their samples.
  await panel.getByLabel('Operation', {exact: true}).selectOption('geometry-overlay-intersection');
  await setAnalysisSheetParameter(panel, 'coordinateSystem', 'EPSG:4326');
  await panel.getByTestId('catalog-run').click();
  await page.waitForFunction(() =>
    /GPUPolygonOverlay/.test(
      document.querySelector('[data-testid="catalog-engine"]')?.textContent ?? ''
    )
  );
  assert.match(
    await panel.getByTestId('catalog-result').innerText(),
    /Overlay intersection: [1-9]\d* rows/
  );
  await panel.getByLabel('Operation', {exact: true}).selectOption('network-crossing-noding');
  await setAnalysisSheetParameter(panel, 'coordinateSystem', 'EPSG:4326');
  await panel.getByTestId('catalog-run').click();
  await page.waitForFunction(() =>
    /GPUNetworkNoding/.test(
      document.querySelector('[data-testid="catalog-engine"]')?.textContent ?? ''
    )
  );
  await panel.getByLabel('Operation', {exact: true}).selectOption('network-service-area');
  await setAnalysisSheetParameter(panel, 'coordinateSystem', 'planar');
  await setAnalysisSheetParameter(panel, 'onewayProperty', 'oneway');
  await setAnalysisSheetParameter(panel, 'maximumSnapDistance', 1);
  await setAnalysisSheetParameter(panel, 'budgets', [15]);

  // Serialization round trip restores parameters.
  await panel.getByRole('button', {name: 'Serialize invocation', exact: true}).click();
  const serialized = JSON.parse(
    await panel.getByLabel('Serialized invocation', {exact: true}).inputValue()
  );
  assert.equal(serialized.operationId, 'network-service-area');
  assert.deepEqual(serialized.parameters.budgets, [15]);
  assert.equal(typeof serialized.inputs.lines.contentHash, 'string');
  await setAnalysisSheetParameter(panel, 'budgets', [99]);
  await panel.getByRole('button', {name: 'Restore parameters', exact: true}).click();
  assert.equal(await panel.getByTestId('catalog-parameter-budgets').inputValue(), '15');

  // Shared controls retain invalid text locally and show the parameter's range error.
  await setAnalysisSheetParameter(panel, 'budgets', [-3]);
  await panel.getByTestId('catalog-parameter-error-budgets').waitFor();
  await setAnalysisSheetParameter(panel, 'budgets', [15]);

  // Geometry workflow: convex clip returns new part identities.
  await panel.getByLabel('Operation', {exact: true}).selectOption('geometry-convex-clip');
  await panel.getByTestId('catalog-run').click();
  assert.match(await panel.getByTestId('catalog-result').innerText(), /Clipped features: 1 rows/);

  // Exercise the same complete staging/publication path for the existing polygon clip workflow.
  await publishPolygonResult(panel, 'Clipped features', 1);
  assert.deepEqual(failures, []);
  await closeNativeDialog(page);

  // Preserve numeric epoch seconds through Prepare data. The native Kepler CSV analyzer
  // infers these integers as timestamps, whose original-view readback is ISO text; the
  // spacetime bindings explicitly accept numeric epochs only.
  // Strip the checked-in fixture's two SPDX comment lines before passing it to the CSV loader.
  const fixture = await readFile(new URL('../data/spacetime-observations.csv', import.meta.url), 'utf8');
  await openNativeData(page);
  const csvWorkspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await csvWorkspace.getByLabel('Open Arrow IPC file', {exact: true}).setInputFiles({
    name: 'spacetime-observations.csv', mimeType: 'text/csv',
    buffer: Buffer.from(fixture.replace(/^#.*\r?\n/gm, ''))
  });
  const csvPreparation = csvWorkspace.getByRole('region', {name: 'Prepare spacetime-observations.csv', exact: true});
  await csvPreparation.waitFor();
  await csvPreparation.getByLabel('Float64 conversion', {exact: true}).selectOption('round');
  await csvPreparation.getByLabel('Maximum absolute error', {exact: true}).fill('0.00001');
  await csvPreparation.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  const previousPublicationIds = await page.evaluate(() =>
    window.__keplerComputeTest.getState().publications.map(source => source.id)
  );
  await csvPreparation.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    knownIds => {
      // This hook exposes IDs and counts; it has no label property. The dataset selector
      // below independently verifies the fixture label before any spacetime operation runs.
      const publication = window.__keplerComputeTest.getState().publications.find(source =>
        !knownIds.includes(source.id)
      );
      if (publication?.error) throw new Error(publication.error);
      return publication?.rowCount === 48 && publication.selectedCount === 48;
    },
    previousPublicationIds,
    {timeout: 60000}
  );
  await closeNativeDialog(page);
  const spacetimeOperations = [
    ['space-time-kernel-density', 'GPUSpaceTimeKernelDensity', /voxelCount: 64/],
    ['space-time-relative-risk', 'GPUSpaceTimeRelativeRisk', /caseCount: 24/],
    ['adaptive-bandwidth-kernel-density', 'GPUAdaptiveBandwidthKernelDensity', /voxelCount: 16/],
    ['space-time-hawkes', 'GPUSpaceTimeHawkes', /branchingRatio:/],
    ['expectation-based-space-time-scan', 'GPUExpectationBasedSpaceTimeScan', /logLikelihoodRatio:/],
    ['space-time-cross-correlation', 'GPUSpaceTimeCrossCorrelation', /testedLocations: 6/],
    ['space-time-local-moran', 'GPUSpaceTimeLocalMoran', /period: 7/],
    ['jacquez-test', 'GPUJacquezTest', /observed:/],
    ['local-knox', 'GPULocalKnox', /observedTotal:/],
    ['near-repeat-knox', 'GPUNearRepeatKnox', /bandCount: 3/],
    ['space-time-diggle-test', 'GPUSpaceTimeDiggleTest', /pseudoPGreater:/],
    ['space-time-k-function', 'GPUSpaceTimeKFunction', /classCount: 3/],
    ['pettitt-test', 'GPUPettittTest', /statisticMaximum:/],
    ['location-change-points', 'GPULocationChangePoints', /changePointCount:/]
  ];
  for (const [operationId, contributor, expectedText] of spacetimeOperations) {
    await panel.getByLabel('Operation', {exact: true}).selectOption(operationId);
    assert.match(await panel.getByTestId('catalog-limits').innerText(), /maximum 512 input rows/);
    await setAnalysisSheetParameter(panel, 'timeUnit', 'seconds');
    await panel.getByLabel('Event points source', {exact: true}).selectOption('native');
    const datasetSelector = panel.getByLabel('Event points dataset', {exact: true});
    const fixtureOption = datasetSelector.locator('option').filter({hasText: /^spacetime-observations(?:\.csv)? \(resident, 48 rows\)$/});
    await fixtureOption.waitFor({state: 'attached', timeout: 15000});
    const fixtureLabel = await fixtureOption.innerText();
    assert.match(fixtureLabel, /spacetime-observations.*\(resident, 48 rows\)/);
    await datasetSelector.selectOption({label: fixtureLabel});
    await panel.getByLabel('Event points geometry field', {exact: true}).selectOption({label: 'Point from numeric columns'});
    await panel.getByLabel('Event points X / longitude column', {exact: true}).selectOption('longitude');
    await panel.getByLabel('Event points Y / latitude column', {exact: true}).selectOption('latitude');
    await panel.getByTestId('catalog-check-inputs').click();
    await setAnalysisSheetParameter(panel, 'timeProperty', 'event_seconds');
    await setAnalysisSheetParameter(panel, 'permutations', 3, {optional: true});
    await setAnalysisSheetParameter(panel, 'maximumIterations', 3, {optional: true});
    await panel.getByTestId('catalog-run').click();
    await page.waitForFunction(name => document.querySelector('[data-testid="catalog-engine"]')?.textContent?.includes(name) || document.querySelector('[data-testid="catalog-error"]'), contributor, {timeout: 60000});
    assert.equal(await panel.getByTestId('catalog-error').count(), 0, `${operationId} succeeds`);
    assert.match(await panel.getByTestId('catalog-engine').innerText(), new RegExp(`Engine: GPU.*${contributor}`));
    const output = await panel.getByTestId('catalog-result').innerText();
    assert.match(output, /featureCount: 48/);
    assert.match(output, expectedText);
    if (operationId === 'location-change-points') {
      assert.match(output, /Spacetime point results: 48 rows/);
      assert.match(output, /source_index/);
      await panel.getByTestId('catalog-publish').click();
      await openNativeData(page);
      await page
        .getByRole('region', {name: 'Resident datasets', exact: true})
        .getByText('Spacetime point results').first().waitFor({timeout: 15000});
      await closeNativeDialog(page);
    }
  }
  assert.deepEqual(failures, []);
  await exerciseTrajectoryCatalog(page, panel);
  await checkPointCloudCatalogWorkflow(page, panel);
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({ok: true, spacetimeOperations: spacetimeOperations.length}));
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
