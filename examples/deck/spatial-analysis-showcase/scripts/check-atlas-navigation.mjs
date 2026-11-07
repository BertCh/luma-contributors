// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';

const baseUrl = process.argv.find(argument => argument.startsWith('--url='))?.slice(6) ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv.find(argument => argument.startsWith('--output='))?.slice(9) ?? '/tmp/showcase-atlas-navigation';
await mkdir(outputDirectory, {recursive: true});
const buildEvidence = await (await fetch(new URL('build-evidence.json', baseUrl))).json();
await writeFile(`${outputDirectory}/build-evidence.json`, `${JSON.stringify(buildEvidence, null, 2)}\n`);
const browser = await chromium.launch({headless: true});
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const evidence = {bundleSha256: buildEvidence.bundleSha256};
try {
  await page.addInitScript(() => {
    globalThis.__atlasAdapterRequests = 0;
    if (!navigator.gpu) return;
    const requestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
    navigator.gpu.requestAdapter = (...arguments_) => {
      globalThis.__atlasAdapterRequests++;
      return requestAdapter(...arguments_);
    };
  });
  await page.goto(baseUrl, {waitUntil: 'networkidle'});
  await page.locator('#atlas-search').waitFor();
  evidence.startup = await page.evaluate(() => {
    const scripts = performance.getEntriesByType('resource').filter(entry => entry.name.endsWith('.js'));
    return {
      scriptRequests: scripts.length,
      decodedScriptBytes: scripts.reduce((total, entry) => total + entry.decodedBodySize, 0),
      loadedGpuHost: scripts.some(entry => /\/deck-host-[^/]+\.js$/.test(entry.name)),
      loadedComputeModule: scripts.some(entry => /\/[^/]+\.compute-[^/]+\.js$/.test(entry.name)),
      loadedRecordedFigures: scripts.some(entry => /\/lens-twins-[^/]+\.js$/.test(entry.name)),
      adapterRequests: globalThis.__atlasAdapterRequests,
      studyCount: document.querySelector('.atlas-count')?.textContent
    };
  });
  assert.equal(evidence.startup.loadedGpuHost, false);
  assert.equal(evidence.startup.loadedComputeModule, false);
  assert.equal(evidence.startup.loadedRecordedFigures, false);
  assert.equal(evidence.startup.adapterRequests, 0);

  await page.locator('#atlas-search').fill('GPURasterSeasonalBreak');
  await page.waitForFunction(() => document.querySelector('.atlas-count')?.textContent?.startsWith('1 of '));
  const filteredLocation = page.url();
  await page.reload({waitUntil: 'networkidle'});
  assert.equal(await page.locator('#atlas-search').inputValue(), 'GPURasterSeasonalBreak');
  await page.locator('#atlas-search').fill('no-such-place-xyz');
  await page.waitForFunction(() => document.querySelector('.atlas-count')?.textContent?.startsWith('0 of '));
  await page.getByRole('button', {name: 'Clear filters'}).click();
  assert.equal(await page.locator('#atlas-search').inputValue(), '');
  evidence.search = {filteredLocation, reloadAndEmptyReset: true};

  // Abort navigation while the route code is still loading. A late import must
  // neither remount a story nor create a GPU device after returning to the atlas.
  await page.route('**/story-view-*.js', async route => {
    await new Promise(resolve => setTimeout(resolve, 600));
    await route.continue();
  });
  const storyRequest = page.waitForRequest(request => /\/story-view-[^/]+\.js$/.test(request.url()));
  await page.evaluate(() => { location.hash = '#/story/rail-resilience'; });
  await storyRequest;
  await page.evaluate(() => { location.hash = '#/'; });
  await page.locator('#atlas-search').waitFor();
  await page.waitForTimeout(900);
  evidence.cancelledStory = await page.evaluate(() => ({
    atlasVisible: Boolean(document.querySelector('#atlas-search')),
    storyClass: document.body.classList.contains('is-story'),
    adapterRequests: globalThis.__atlasAdapterRequests
  }));
  assert.equal(evidence.cancelledStory.atlasVisible, true);
  assert.equal(evidence.cancelledStory.storyClass, false);
  assert.equal(evidence.cancelledStory.adapterRequests, 0);

  await page.route('**/lens-twins-*.js', async route => {
    await new Promise(resolve => setTimeout(resolve, 600));
    await route.continue();
  });
  const referenceRequest = page.waitForRequest(request => /\/lens-twins-[^/]+\.js$/.test(request.url()));
  await page.evaluate(() => { location.hash = '#/reference/GPUPointDensity'; });
  await referenceRequest;
  await page.evaluate(() => { location.hash = '#/'; });
  await page.locator('#atlas-search').waitFor();
  await page.waitForTimeout(1100);
  assert.equal(await page.locator('#atlas-search').count(), 1);
  evidence.cancelledReference = true;
  await page.evaluate(() => { location.hash = '#/reference/GPUPointDensity'; });
  await page.locator('.doc').first().waitFor();
  assert.equal(await page.locator('.admonition-danger').count(), 0);
  evidence.reference = {recordedFigures: await page.locator('figure.lens-twin').count()};
  assert.ok(evidence.reference.recordedFigures > 0, 'Reference should still render recorded figures.');

  await page.evaluate(() => { location.hash = '#/'; });
  await page.locator('#atlas-search').waitFor();
  await page.setViewportSize({width: 320, height: 800});
  await page.waitForTimeout(100);
  evidence.narrowLayout = await page.evaluate(() => ({width: innerWidth, documentWidth: document.documentElement.scrollWidth}));
  assert.equal(evidence.narrowLayout.documentWidth, evidence.narrowLayout.width);
  await page.locator('.skip-link').focus();
  await page.keyboard.press('Enter');
  assert.equal(await page.evaluate(() => document.activeElement?.id), 'view');
  await page.screenshot({path: `${outputDirectory}/atlas-mobile.png`});
  await page.setViewportSize({width: 1440, height: 1000});
  await page.screenshot({path: `${outputDirectory}/atlas-desktop.png`});
  await page.evaluate(() => { location.hash = '#/data'; });
  await page.locator('#data-search').waitFor();
  const datasetCount = await page.locator('.dataset-card').count();
  assert.ok(datasetCount > 0);
  await page.locator('#data-search').fill('Hochebenkar');
  assert.equal(await page.locator('.dataset-card').count(), 1);
  await page.reload({waitUntil: 'networkidle'});
  assert.equal(await page.locator('#data-search').inputValue(), 'Hochebenkar');
  assert.equal(await page.locator('.dataset-card').count(), 1);
  await page.locator('.dataset-source summary').click();
  assert.equal(await page.locator('.dataset-bounds:visible').count(), 1);
  assert.ok(await page.getByRole('link', {name: 'Read the source'}).getAttribute('href'));
  await page.screenshot({path: `${outputDirectory}/data-desktop.png`});
  await page.locator('#data-search').fill('no-such-dataset-xyz');
  assert.equal(await page.locator('.dataset-card').count(), 0);
  await page.getByRole('button', {name: 'Clear filters'}).click();
  assert.equal(await page.locator('.dataset-card').count(), datasetCount);
  await page.locator('#data-usage').selectOption('used');
  assert.equal(await page.getByText('Available for a new investigation', {exact: true}).count(), 0);
  await page.setViewportSize({width: 320, height: 800});
  await page.screenshot({path: `${outputDirectory}/data-mobile.png`});
  const dataWidth = await page.evaluate(() => ({width: innerWidth, documentWidth: document.documentElement.scrollWidth}));
  assert.equal(dataWidth.documentWidth, dataWidth.width);
  evidence.datasets = {datasetCount, searchPersists: true, sourceDetails: true, usageFilter: true, narrowLayout: dataWidth};
  assert.equal(await page.evaluate(() => globalThis.__atlasAdapterRequests), 0);
  const failedRoutePage = await browser.newPage();
  const failedRouteErrors = [];
  failedRoutePage.on('pageerror', error => failedRouteErrors.push(error.message));
  await failedRoutePage.route('**/data-view-*.js', route => route.abort('failed'));
  const failedRouteUrl = new URL(baseUrl);
  failedRouteUrl.hash = '#/data';
  await failedRoutePage.goto(failedRouteUrl.toString());
  await failedRoutePage.getByRole('heading', {name: 'Unable to load this page'}).waitFor();
  await failedRoutePage.getByRole('button', {name: 'Back to stories'}).click();
  await failedRoutePage.locator('#atlas-search').waitFor();
  assert.deepEqual(failedRouteErrors, []);
  evidence.failedRouteRecovery = true;
  await failedRoutePage.close();
  const narrativePage = await browser.newPage();
  narrativePage.on('pageerror', error => errors.push(error.message));
  await narrativePage.addInitScript(() => Object.defineProperty(navigator, 'gpu', {value: undefined}));
  const narrativeUrl = new URL(baseUrl);
  narrativeUrl.hash = '#/story/health-regression';
  await narrativePage.goto(narrativeUrl.toString());
  await narrativePage.locator('.step-list button.step-row').first().waitFor();
  assert.equal(await narrativePage.locator('.story-continuation').count(), 0);
  await narrativePage.locator('.step-list button.step-row').last().click();
  const continuation = narrativePage.getByRole('navigation', {name: 'Continue exploring'});
  await continuation.waitFor();
  const onwardLink = continuation.getByRole('link', {name: 'Interpolate a measured surface'});
  assert.equal(await onwardLink.getAttribute('href'), '#/story/rainfall-interpolation');
  await onwardLink.click();
  await narrativePage.waitForURL('**/#/story/rainfall-interpolation');
  await narrativePage.locator('.step-list button.step-row').first().waitFor();
  await narrativePage.locator('.step-list button.step-row').last().click();
  await narrativePage.getByRole('navigation', {name: 'Continue exploring'}).getByRole('link', {name: 'Browse all investigations'}).click();
  await narrativePage.locator('#atlas-search').waitFor();
  evidence.narrativeContinuationWithoutGPU = true;
  await narrativePage.close();
  const finalBuildEvidence = await (await fetch(new URL('build-evidence.json', baseUrl))).json();
  evidence.bundleUnchanged = finalBuildEvidence.bundleSha256 === buildEvidence.bundleSha256;
  assert.equal(evidence.bundleUnchanged, true, 'The served build changed during navigation checks.');
  assert.deepEqual(errors, []);
  console.log('PASS atlas startup, query state, cancelled route races, recorded reference, keyboard and narrow layout.');
} finally {
  await writeFile(`${outputDirectory}/report.json`, `${JSON.stringify({evidence, errors}, null, 2)}\n`);
  await browser.close();
}
