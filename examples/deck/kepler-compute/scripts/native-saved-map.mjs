// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {withSyntheticSample} from './native-navigation.mjs';

// Browser acceptance for unmodified native Kepler files: raw CSV/GeoJSON uploads and saved
// Kepler map JSON (kepler.json) uploads through the native Add Data > Load Files dialog.
//
// Prerequisites: dev server on the URL (default http://127.0.0.1:5318/), fixtures generated with
// `node scripts/kepler-fixtures/generate.mjs`.
// Usage: node scripts/native-saved-map.mjs [url] [--only=name,name] [--report-only]
//   --only         run the listed case names (see CASES below)
//   --report-only  print per-layer misses of the all-layer-types case without failing

import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
import {getSourceCheckUrl, waitForAutomaticPreparation} from './source-check-utils.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {closeNativeDialog} from './native-navigation.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const directory = path.dirname(fileURLToPath(import.meta.url));
const dataDirectory = path.join(directory, 'kepler-fixtures', 'data');
const urlArgument = process.argv.slice(2).find(argument => argument.startsWith('http'));
const url = getSourceCheckUrl(urlArgument, {rows: 100000});
const onlyArgument = process.argv.find(argument => argument.startsWith('--only='));
const onlyNames = onlyArgument ? onlyArgument.slice('--only='.length).split(',') : null;
const isReportOnly = process.argv.includes('--report-only');
const SETTLE_TIMEOUT = 120000;

/** Raw uploads: the dataset must appear and Kepler's auto-created layers must render. */
const RAW_CASES = [
  {name: 'earthquakes-csv', fileName: 'earthquakes.csv', mimeType: 'text/csv', minimumLayers: 1},
  {name: 'nyc-trips-csv', fileName: 'nyc-trips.csv', mimeType: 'text/csv', minimumLayers: 1},
  {
    name: 'sf-zip-geojson',
    fileName: 'sf-zip-geometry.geojson',
    mimeType: 'application/geo+json',
    minimumLayers: 1
  },
  {name: 'h3-hexagons-csv', fileName: 'h3-hexagons.csv', mimeType: 'text/csv', minimumLayers: 1},
  // Native Kepler's findPointFieldPairs requires a separator around lat/lng: lat0/lng0 are not
  // inferred coordinates, in the old host too. The old check counted the startup point layer.
  // Raw upload is a 400-row table; arcs/flow are tested with explicit roles in arcs-flows-map.
  {
    name: 'arcs-flows-csv',
    fileName: 'arcs-flows.csv',
    mimeType: 'text/csv',
    minimumLayers: 0,
    tableRows: 400
  },
  {
    name: 'trips-geojson',
    fileName: 'trips.geojson',
    mimeType: 'application/geo+json',
    minimumLayers: 1
  }
];
/** Saved maps: every saved layer must render and the native config must round-trip. */
const MAP_CASES = [
  'earthquakes.kepler.json',
  'nyc-trips.kepler.json',
  'sf-zips.kepler.json',
  'h3-hexagons.kepler.json',
  'arcs-flows.kepler.json',
  'trips.kepler.json',
  'all-layer-types.kepler.json'
].map(fileName => ({
  name: fileName.replace('.kepler.json', '-map'),
  fileName,
  mimeType: 'application/json',
  isMap: true
}));
const CASES = [...RAW_CASES, ...MAP_CASES].filter(
  entry => !onlyNames || onlyNames.includes(entry.name)
);

// The icon layer fetches Kepler's hosted catalog; serve a deterministic local one.
const SVG_ICONS = [
  {
    id: 'diamond',
    mesh: {
      positions: [
        [0, -1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0]
      ],
      cells: [
        [0, 1, 2],
        [0, 2, 3]
      ]
    }
  },
  {
    id: 'triangle',
    mesh: {
      positions: [
        [0, -1, 0],
        [1, 1, 0],
        [-1, 1, 0]
      ],
      cells: [[0, 1, 2]]
    }
  },
  {
    id: 'square',
    mesh: {
      positions: [
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0]
      ],
      cells: [
        [0, 1, 2],
        [0, 2, 3]
      ]
    }
  }
];

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const summaries = [];
let hasFailure = false;

try {
  for (const entry of CASES) {
    const summary = await runCase(entry);
    summaries.push(summary);
    if (!summary.passed) hasFailure = true;
    console.log(JSON.stringify(summary));
  }
} finally {
  await browser.close();
}
if (hasFailure) {
  console.error('native-saved-map: one or more cases failed');
  process.exitCode = 1;
} else {
  console.log(JSON.stringify({passed: true, cases: summaries.map(summary => summary.name)}));
}

async function runCase(entry) {
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  await pinViteModules(page);
  await page.route('**/icons/svg-icons.json', route =>
    route.fulfill({json: {svgIcons: SVG_ICONS}})
  );
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (/unresolved value|Compilation error|Error while parsing WGSL/.test(message.text()))
      errors.push(message.text());
  });
  const summary = {name: entry.name, passed: false, failures: []};
  const fail = message => summary.failures.push(message);
  try {
    const buffer = readFileSync(path.join(dataDirectory, entry.fileName));
    const expected = entry.isMap ? readExpectedMap(buffer) : null;
    await page.goto(withSyntheticSample(url), {waitUntil: 'domcontentloaded'});
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
      .setInputFiles({name: entry.fileName, mimeType: entry.mimeType, buffer});
    await page
      .locator('.ReactModal__Content')
      .getByRole('button', {name: 'Add Data', exact: true})
      .click();
    await closeNativeDialog(page);

    await waitForAutomaticPreparation(
      page,
      expected
        ? expected.datasets.map(dataset => dataset.label)
        : [entry.fileName.replace(/\.[^.]+$/, '')],
      SETTLE_TIMEOUT
    );

    if (expected) await checkSavedMap(page, entry, expected, summary, fail);
    else await checkRawUpload(page, entry, summary, fail, baselineLayerIds);

    // The native "Prepare data" notice means the dataset stayed unprepared; auto-prepare must not leave it.
    summary.noticeCount = await page.getByTestId('resident-workflow-notice').count();
    if (summary.noticeCount)
      fail(
        `resident-workflow-notice present: ${await page.getByTestId('resident-workflow-notice').first().innerText()}`
      );
    const state = await page.evaluate(() => window.__keplerComputeTest.getState());
    if (state.error) fail(`host error: ${state.error}`);
    summary.publicationIds = state.publications.map(publication => publication.id);
    if (errors.length) fail(`console/page errors: ${errors.slice(0, 3).join(' | ')}`);
    await page.screenshot({path: `/tmp/luma-kepler-saved-${entry.name}.png`});
  } catch (error) {
    fail(`exception: ${error instanceof Error ? error.message : String(error)}`);
    summary.bodyText = (
      await page
        .locator('body')
        .innerText()
        .catch(() => '')
    ).slice(0, 600);
  } finally {
    await page.close();
  }
  summary.passed = summary.failures.length === 0;
  return summary;
}

function readExpectedMap(buffer) {
  const saved = JSON.parse(buffer.toString('utf8'));
  const visState = saved.config.config.visState;
  return {
    datasets: saved.datasets.map(dataset => ({
      id: dataset.data.id,
      label: dataset.data.label,
      rows: dataset.data.allData.length
    })),
    layers: visState.layers.map(layer => ({
      id: layer.id,
      type: layer.type,
      dataId: layer.config.dataId,
      columns: layer.config.columns
    })),
    filters: visState.filters.map(filter => ({
      id: filter.id,
      type: filter.type,
      name: filter.name,
      dataId: filter.dataId
    })),
    tooltip: Object.fromEntries(
      Object.entries(visState.interactionConfig.tooltip.fieldsToShow).map(([dataId, fields]) => [
        dataId,
        fields.map(field => field.name)
      ])
    )
  };
}

async function readRenderedLayerIds(page) {
  return page.evaluate(() =>
    window.__keplerComputeTest
      .renderedLayers()
      .filter(layer => layer.loaded && layer.visible)
      .map(layer => layer.id)
  );
}

async function checkRawUpload(page, entry, summary, fail, baselineLayerIds) {
  const label = entry.fileName;
  await page
    .getByText(label, {exact: false})
    .first()
    .waitFor({timeout: 60000})
    .catch(() => fail(`dataset label ${label} never appeared`));
  const config = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  // Startup already has a visible point layer, in the old example as well. Counting it let a
  // failed upload pass this rendering check without creating a single new layer.
  const nativeLayers = config.config.visState.layers.filter(
    layer => !baselineLayerIds.includes(layer.id)
  );
  summary.nativeLayers = nativeLayers.map(layer => `${layer.type}:${layer.config.dataId}`);
  if (entry.tableRows !== undefined) {
    assert.equal(nativeLayers.length, 0, 'Unrecognized coordinate names must stay table-only');
    const publications = await page.evaluate(
      () => window.__keplerComputeTest.getState().publications
    );
    assert.equal(publications.length, 1);
    assert.equal(publications[0].rowCount, entry.tableRows);
    assert.equal(publications[0].selectedCount, entry.tableRows);
    summary.renderedLayerIds = await readRenderedLayerIds(page);
    return;
  }
  if (nativeLayers.length < entry.minimumLayers)
    fail(`Kepler created ${nativeLayers.length} layers, expected at least ${entry.minimumLayers}`);
  try {
    await page.waitForFunction(
      ({layerIds, minimum}) =>
        window.__keplerComputeTest
          .renderedLayers()
          .filter(
            layer => layer.loaded && layer.visible && layerIds.some(id => layer.id.includes(id))
          ).length >= minimum,
      {layerIds: nativeLayers.map(layer => layer.id), minimum: entry.minimumLayers},
      {timeout: SETTLE_TIMEOUT}
    );
  } catch {
    fail('no loaded visible rendered layers');
  }
  summary.renderedLayerIds = await readRenderedLayerIds(page);
}

async function checkSavedMap(page, entry, expected, summary, fail) {
  for (const dataset of expected.datasets) {
    await page
      .getByText(dataset.label, {exact: false})
      .first()
      .waitFor({timeout: 60000})
      .catch(() => fail(`dataset ${dataset.id} (${dataset.label}) never appeared`));
  }
  // Native round trip: layers, filters and tooltip must survive the load unchanged.
  const config = (await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig())).config
    .visState;
  const nativeLayers = new Map(config.layers.map(layer => [layer.id, layer]));
  for (const layer of expected.layers) {
    const loaded = nativeLayers.get(layer.id);
    if (!loaded) {
      fail(`layer ${layer.id} (${layer.type}) missing from native state`);
      continue;
    }
    if (loaded.type !== layer.type) fail(`layer ${layer.id} type ${loaded.type} != ${layer.type}`);
    if (loaded.config.dataId !== layer.dataId)
      fail(`layer ${layer.id} dataId ${loaded.config.dataId} != ${layer.dataId}`);
    try {
      assert.deepEqual(loaded.config.columns, layer.columns);
    } catch {
      fail(`layer ${layer.id} columns changed: ${JSON.stringify(loaded.config.columns)}`);
    }
  }
  const loadedFilters = new Map(config.filters.map(filter => [filter.id, filter]));
  for (const filter of expected.filters) {
    const loaded = loadedFilters.get(filter.id);
    if (!loaded || loaded.type !== filter.type)
      fail(`filter ${filter.id} (${filter.type}) missing or retyped`);
    else if (JSON.stringify(loaded.name) !== JSON.stringify(filter.name))
      fail(`filter ${filter.id} fields changed`);
  }
  for (const [dataId, names] of Object.entries(expected.tooltip)) {
    const loaded = config.interactionConfig.tooltip.fieldsToShow[dataId]?.map(field => field.name);
    if (JSON.stringify(loaded) !== JSON.stringify(names))
      fail(`tooltip fields for ${dataId} changed: ${JSON.stringify(loaded)}`);
  }

  // Rendering: every saved layer id must be backed by a loaded, visible Deck layer. Resident layers
  // may carry host suffixes, so a rendered id matches when it contains the saved layer id.
  let renderedIds = [];
  const isRendered = layer => renderedIds.some(renderedId => renderedId.includes(layer.id));
  const deadline = Date.now() + SETTLE_TIMEOUT;
  while (Date.now() < deadline) {
    renderedIds = await readRenderedLayerIds(page);
    if (expected.layers.every(isRendered)) break;
    await page.waitForTimeout(1000);
  }
  summary.renderedLayerIds = renderedIds;
  const missing = expected.layers.filter(layer => !isRendered(layer));
  summary.missingLayers = missing.map(layer => `${layer.id}:${layer.type}`);
  if (missing.length && !(isReportOnly && entry.name === 'all-layer-types-map')) {
    fail(`saved layers not rendered: ${summary.missingLayers.join(', ')}`);
  }
}
