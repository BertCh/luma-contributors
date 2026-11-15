// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Downloads Kepler's public sample datasets (keplergl/kepler.gl-data on GitHub) into the gitignored
// scripts/kepler-demo-cache/ directory and loads each one through native Add Data > Load Files,
// asserting the dataset appears, auto-prepare reports an outcome, there are no WGSL errors and no
// resident-workflow-notice. Network failures never fail the run: missing files are skipped, and a
// fully offline run exits 0 with "SKIPPED (offline)".
//
// Prerequisites for the browser part: dev server on the URL (default http://127.0.0.1:5318/).
// Usage: node scripts/kepler-demo-files.mjs [url] [--download-only] [--only=name,name] [--max-bytes=N]

import assert from 'node:assert/strict';
import {dsvFormat} from 'd3-dsv';
import {getSourceCheckUrl, waitForAutomaticPreparation} from './source-check-utils.mjs';
import {existsSync, mkdirSync, readFileSync, statSync, writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const cacheDirectory = path.join(directory, 'kepler-demo-cache');
const BASE_URL = 'https://raw.githubusercontent.com/keplergl/kepler.gl-data/master/';
const DOWNLOAD_TIMEOUT_MS = 20000;
const SETTLE_TIMEOUT = 120000;

const argumentList = process.argv.slice(2);
const url = getSourceCheckUrl(
  argumentList.find(argument => argument.startsWith('http://') || argument.startsWith('https://')),
  {rows: 100000}
);
const isDownloadOnly = argumentList.includes('--download-only');
const onlyArgument = argumentList.find(argument => argument.startsWith('--only='));
const onlyNames = onlyArgument ? onlyArgument.slice('--only='.length).split(',') : null;
const maxBytesArgument = argumentList.find(argument => argument.startsWith('--max-bytes='));
const maxBytes = maxBytesArgument
  ? Number(maxBytesArgument.slice('--max-bytes='.length))
  : 15_000_000;

/** Paths inside keplergl/kepler.gl-data (verified against the GitHub tree listing). */
const DATASETS = [
  {name: 'earthquakes', remotePath: 'earthquakes/data.csv', mimeType: 'text/csv'},
  {name: 'nyctrips', remotePath: 'nyctrips/data.csv', mimeType: 'text/csv'},
  {name: 'sftrees', remotePath: 'sftrees/data.csv', mimeType: 'text/csv'},
  {name: 'movement-pittsburgh', remotePath: 'movement_pittsburgh/data.csv', mimeType: 'text/csv'},
  {
    name: 'county-unemployment',
    remotePath: 'county_unemployment/data.geojson',
    mimeType: 'application/geo+json'
  },
  {
    name: 'buildings-australia',
    remotePath: 'datasets/buildings-australia.geojson',
    mimeType: 'application/geo+json'
  }
].filter(entry => !onlyNames || onlyNames.includes(entry.name));

mkdirSync(cacheDirectory, {recursive: true});
const available = [];
const skipped = [];
for (const entry of DATASETS) {
  const result = await fetchDataset(entry);
  if (result.filePath) available.push({...entry, filePath: result.filePath, source: result.source});
  else skipped.push({name: entry.name, reason: result.reason});
}
console.log(
  JSON.stringify({
    download: {available: available.map(({name, source}) => ({name, source})), skipped}
  })
);

if (!available.length) {
  console.log(
    'SKIPPED (offline): no Kepler sample datasets could be downloaded or found in the cache'
  );
  process.exit(0);
}
if (isDownloadOnly) process.exit(0);

const {chromium} = await import('playwright');
const {getPlaywrightLaunchOptions} = await import(
  '../../../../scripts/playwright/get-playwright-launch-options.mjs'
);
const {closeNativeDialog} = await import('./native-navigation.mjs');
const {pinViteModules} = await import('./pin-vite-modules.mjs');

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
let hasFailure = false;
try {
  for (const entry of available) {
    const summary = await runCase(entry);
    if (!summary.passed) hasFailure = true;
    console.log(JSON.stringify(summary));
  }
} finally {
  await browser.close();
}
if (hasFailure) {
  console.error('kepler-demo-data: one or more datasets failed');
  process.exitCode = 1;
} else {
  console.log(
    JSON.stringify({passed: true, datasets: available.map(entry => entry.name), skipped})
  );
}

async function fetchDataset(entry) {
  const filePath = path.join(cacheDirectory, `${entry.name}${path.extname(entry.remotePath)}`);
  if (existsSync(filePath) && statSync(filePath).size > 0) return {filePath, source: 'cache'};
  try {
    const response = await fetch(BASE_URL + entry.remotePath, {
      signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS)
    });
    if (!response.ok) return {reason: `HTTP ${response.status}`};
    const declaredBytes = Number(response.headers.get('content-length') || 0);
    if (declaredBytes > maxBytes) return {reason: `larger than --max-bytes (${declaredBytes})`};
    const buffer = Buffer.from(await response.arrayBuffer());
    if (buffer.byteLength > maxBytes)
      return {reason: `larger than --max-bytes (${buffer.byteLength})`};
    writeFileSync(filePath, buffer);
    return {filePath, source: 'network'};
  } catch (error) {
    return {reason: `network: ${error instanceof Error ? error.message : String(error)}`};
  }
}

function describeFile(buffer, entry) {
  if (entry.mimeType === 'text/csv') {
    // Demo CSV includes quoted geometry and text: physical lines/commas are not records/fields.
    const records = dsvFormat(',').parseRows(buffer.toString('utf8'));
    return {rows: records.length - 1, columns: records[0].length};
  }
  const features = JSON.parse(buffer.toString('utf8')).features || [];
  return {rows: features.length, columns: Object.keys(features[0]?.properties || {}).length + 1};
}

async function runCase(entry) {
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  await pinViteModules(page);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (/unresolved value|Compilation error|Error while parsing WGSL/.test(message.text()))
      errors.push(message.text());
  });
  const summary = {
    name: entry.name,
    passed: false,
    rows: null,
    columns: null,
    layers: null,
    outcome: null,
    failures: []
  };
  const fail = message => summary.failures.push(message);
  try {
    const buffer = readFileSync(entry.filePath);
    Object.assign(summary, describeFile(buffer, entry));
    const fileName = path.basename(entry.remotePath).replace(/^data\./, `${entry.name}.`);
    await page.goto(url, {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(
      () => {
        const state = window.__keplerComputeTest?.getState();
        if (state?.error) throw new Error(state.error);
        return state?.selectedCount === 100000;
      },
      undefined,
      {timeout: 60000}
    );
    const baselineLayerIds = await page.evaluate(() =>
      window.__keplerComputeTest.getState().nativeLayers.map(layer => layer.id)
    );
    await page.getByRole('button', {name: 'Add Data', exact: true}).click();
    await page
      .locator('.ReactModal__Content input[type=file]')
      .setInputFiles({name: fileName, mimeType: entry.mimeType, buffer});
    await page
      .locator('.ReactModal__Content')
      .getByRole('button', {name: 'Add Data', exact: true})
      .click();
    await closeNativeDialog(page);
    await page
      .getByText(fileName, {exact: false})
      .first()
      .waitFor({timeout: 60000})
      .catch(() => fail(`dataset ${fileName} never appeared`));
    const report = page.getByTestId('auto-prepare-report');
    // A report appears while queued/preparing too. The old check could pass on that prefix even
    // when publication subsequently failed; require the final Prepared outcome before config checks.
    await waitForAutomaticPreparation(page, [fileName.replace(/\.[^.]+$/, '')], SETTLE_TIMEOUT);
    if (await report.count())
      summary.outcome = (await report.innerText()).replace(/\s+/g, ' ').slice(0, 300);
    const config = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    const uploadedLayers = config.config.visState.layers.filter(
      layer => !baselineLayerIds.includes(layer.id)
    );
    summary.layers = uploadedLayers.length;
    assert.ok(summary.layers > 0, 'The demo upload must create its own native layers');
    const visibleLayerIds = uploadedLayers
      .filter(layer => layer.config.isVisible !== false)
      .map(layer => layer.id);
    assert.ok(visibleLayerIds.length > 0, 'The demo upload must have visible layers');
    await page.waitForFunction(
      ids => {
        const rendered = window.__keplerComputeTest.renderedLayers();
        return ids.every(id =>
          rendered.some(layer => layer.id.includes(id) && layer.loaded && layer.visible)
        );
      },
      visibleLayerIds,
      {timeout: SETTLE_TIMEOUT}
    );
    // Every layer must stay visible except arc and line layers, which native Kepler creates hidden by default
    // (layer-utils findDefaultLayers); other geometry columns are prepared on demand as variants.
    summary.hiddenLayers = uploadedLayers.filter(
      layer => layer.config?.isVisible === false && layer.type !== 'arc' && layer.type !== 'line'
    ).length;
    if (summary.hiddenLayers) fail(`${summary.hiddenLayers} layers hidden`);
    if (entry.name === 'movement-pittsburgh') {
      await page.waitForTimeout(3000);
      summary.geometryVariants = await page.evaluate(() =>
        window.__keplerComputeTest.getGeometryVariantState?.()
      );
      if (!summary.geometryVariants?.variantCount)
        fail('no geometry column variants were prepared');
    }
    if (await page.getByTestId('resident-workflow-notice').count())
      fail('resident-workflow-notice present');
    const state = await page.evaluate(() => window.__keplerComputeTest.getState());
    if (state.error) fail(`host error: ${state.error}`);
    if (errors.length) fail(`console/page errors: ${errors.slice(0, 3).join(' | ')}`);
  } catch (error) {
    fail(`exception: ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    await page.close();
  }
  summary.passed = summary.failures.length === 0;
  return summary;
}
