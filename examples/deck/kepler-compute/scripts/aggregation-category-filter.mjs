// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// LAYERS5-2: categorical (mode) color on grid and hexagon layers with a range filter that hides the
// first rows of the bins must color the cells like the native CPU layer built from the surviving
// rows. Runs a small dictionary and a 20,000-label dictionary. Headless WebGPU only.
// Usage: node aggregation-category-filter.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {getFixtureUrl} from './script-fixture-url.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const baseUrl = getFixtureUrl(process.argv[2], {rows: '256', autoPrepare: '0'});
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(60000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (
    message.type() === 'error' &&
    /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())
  )
    errors.push(message.text());
});

function makeRows(rowCount, categoryCount) {
  let seed = 12345;
  const random = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  const rows = [];
  for (let row = 0; row < rowCount; row++) {
    const longitude = -74.08 + random() * 0.16;
    const latitude = 40.68 + random() * 0.14;
    // Spatially coherent category with 4% noise so every cell has a unique mode (mode ties resolve by
    // dictionary code, which legitimately differs between datasets with different dictionaries); a rank column drives the filter.
    const block =
      Math.floor((longitude + 74.08) / 0.02) * 9 + Math.floor((latitude - 40.68) / 0.02);
    const category =
      (block * 7 + (random() < 0.04 ? Math.floor(random() * categoryCount) : 0)) % categoryCount;
    rows.push({
      longitude: Math.fround(longitude),
      latitude: Math.fround(latitude),
      kind: `label-${category}`,
      rank: row
    });
  }
  return rows;
}

async function publishRows(rows, name) {
  const table = new arrow.Table({
    longitude: arrow.vectorFromArray(
      rows.map(row => row.longitude),
      new arrow.Float32()
    ),
    latitude: arrow.vectorFromArray(
      rows.map(row => row.latitude),
      new arrow.Float32()
    ),
    rank: arrow.vectorFromArray(
      rows.map(row => row.rank),
      new arrow.Float32()
    ),
    kind: arrow.vectorFromArray(rows.map(row => row.kind))
  });
  const before = await page.evaluate(
    () => window.__keplerComputeTest.getState().publications.length
  );
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: `${name}.arrow`,
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))
  });
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    ({count, rowCount}) =>
      window.__keplerComputeTest.getState().publications[count]?.selectedCount === rowCount,
    {count: before, rowCount: rows.length}
  );
  await closeNativeDialog(page);
  return page.evaluate(
    count => window.__keplerComputeTest.getState().publications[count].id,
    before
  );
}

function configureLayer(layer, id, dataId, type) {
  layer.id = id;
  layer.type = type;
  layer.config.dataId = dataId;
  layer.config.columnMode = 'points';
  layer.config.columns = {lat: 'latitude', lng: 'longitude'};
  layer.config.colorField = {name: 'kind', type: 'string'};
  layer.config.sizeField = null;
  layer.config.colorScale = 'ordinal';
  layer.config.textLabel = [];
  layer.config.isVisible = true;
  layer.visualChannels = {
    ...layer.visualChannels,
    colorField: layer.config.colorField,
    colorScale: 'ordinal',
    sizeField: null
  };
  layer.config.visConfig = {
    ...layer.config.visConfig,
    worldUnitSize: 1.2,
    resolution: 8,
    coverage: 1,
    enable3d: false,
    percentile: [0, 100],
    opacity: 1,
    colorAggregation: 'mode',
    colorRange: {
      name: 'categorical',
      type: 'qualitative',
      category: 'Uber',
      colors: [
        '#12939A',
        '#DDB27C',
        '#88572C',
        '#FF991F',
        '#F15C17',
        '#223F9A',
        '#DA70BF',
        '#125C77',
        '#4DC19C',
        '#776E57',
        '#17B8BE',
        '#F6D18A'
      ]
    }
  };
  return layer;
}

async function renderAndCapture(configuration, layerId) {
  await page.evaluate(
    config => window.__keplerComputeTest.applyNativeConfig(config),
    configuration
  );
  return captureLayer(layerId);
}

async function captureLayer(layerId) {
  await page.waitForFunction(id => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    const invalid = state.nativeLayers?.find(
      candidate => candidate.id === id && candidate.valid === false
    );
    if (invalid) throw new Error(`${id}: ${JSON.stringify(invalid)} ${state.error ?? ''}`);
    return api
      .renderedLayers()
      .some(candidate => candidate.id === id && candidate.visible && candidate.loaded);
  }, layerId);
  await page.waitForTimeout(1500);
  return PNG.sync.read(await page.screenshot());
}

function comparePixels(left, right) {
  let drawn = 0;
  let different = 0;
  for (let y = 120; y < 880; y++)
    for (let x = 380; x < 1060; x++) {
      const offset = (y * left.width + x) * 4;
      const leftColored =
        Math.max(left.data[offset], left.data[offset + 1], left.data[offset + 2]) -
          Math.min(left.data[offset], left.data[offset + 1], left.data[offset + 2]) >
        40;
      const rightColored =
        Math.max(right.data[offset], right.data[offset + 1], right.data[offset + 2]) -
          Math.min(right.data[offset], right.data[offset + 1], right.data[offset + 2]) >
        40;
      if (leftColored || rightColored) drawn++;
      if (
        [0, 1, 2].some(
          channel => Math.abs(left.data[offset + channel] - right.data[offset + channel]) > 10
        )
      )
        different++;
    }
  return {drawn, different};
}

const results = [];
try {
  await page.goto(baseUrl, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  for (const {categoryCount, rowCount} of [
    {categoryCount: 9, rowCount: 9000},
    {categoryCount: 20000, rowCount: 40000}
  ]) {
    const rows = makeRows(rowCount, categoryCount);
    const hiddenRows = Math.floor(rowCount / 3);
    const survivors = rows.filter(row => row.rank >= hiddenRows);
    const datasetId = await publishRows(rows, `categories-${categoryCount}`);
    // Oracle: the host renders only resident layers (a native Kepler layer on a CPU dataset never
    // reaches the Deck layer list). Native aggregation keeps every point in its bins, numbers the
    // ordinal colors by bin visit order (first row of each bin, filtered or not) and applies the
    // range filter inside each bin's accessor. The same semantics without a filter is a resident
    // dataset with all rows whose hidden rows have a null category: bins and their first rows are
    // unchanged and the mode only sees the surviving rows.
    const survivorsId = await publishRows(
      rows.map(row => (row.rank < hiddenRows ? {...row, kind: null} : row)),
      `categories-${categoryCount}-nulled`
    );
    await page.evaluate(
      ({id, minimum, maximum}) =>
        window.__keplerComputeTest.setPublishedRange(id, 'rank', minimum, maximum),
      {id: datasetId, minimum: hiddenRows, maximum: rowCount}
    );
    await page.waitForFunction(
      ({id, count}) =>
        window.__keplerComputeTest.getState().publications.find(candidate => candidate.id === id)
          ?.selectedCount === count,
      {id: datasetId, count: survivors.length}
    );
    for (const type of ['grid', 'hexagon']) {
      const base = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
      assert(
        base.config.visState.layers.find(layer => layer.config.dataId === datasetId),
        'resident layer exists'
      );
      const residentConfig = structuredClone(base);
      const survivorsTemplate = base.config.visState.layers.find(
        layer => layer.config.dataId === survivorsId
      );
      assert(survivorsTemplate, 'oracle dataset layer exists');
      for (const layer of residentConfig.config.visState.layers) layer.config.isVisible = false;
      const residentLayer = residentConfig.config.visState.layers.find(
        layer => layer.config.dataId === datasetId
      );
      configureLayer(residentLayer, `category-${type}-${categoryCount}-resident`, datasetId, type);
      residentConfig.config.mapState = {
        ...residentConfig.config.mapState,
        longitude: -74,
        latitude: 40.75,
        zoom: 10.8,
        pitch: 0,
        bearing: 0
      };
      const residentImage = await renderAndCapture(residentConfig, residentLayer.id);
      const oracleConfig = structuredClone(base);
      for (const layer of oracleConfig.config.visState.layers) layer.config.isVisible = false;
      const oracleLayer = oracleConfig.config.visState.layers.find(
        layer => layer.config.dataId === survivorsId
      );
      configureLayer(oracleLayer, `category-${type}-${categoryCount}-nulled`, survivorsId, type);
      oracleConfig.config.mapState = residentConfig.config.mapState;
      const nativeImage = await renderAndCapture(oracleConfig, oracleLayer.id);
      if (process.env.CATEGORY_DUMP) {
        const {writeFileSync} = await import('node:fs');
        writeFileSync(
          `${process.env.CATEGORY_DUMP}-${type}-${categoryCount}-resident.png`,
          PNG.sync.write(residentImage)
        );
        writeFileSync(
          `${process.env.CATEGORY_DUMP}-${type}-${categoryCount}-survivors.png`,
          PNG.sync.write(nativeImage)
        );
      }
      const {drawn, different} = comparePixels(residentImage, nativeImage);
      results.push({type, categoryCount, drawn, different});
      console.log(JSON.stringify(results.at(-1)));
      assert(drawn > 1500, `${type}/${categoryCount} must draw cells (${drawn})`);
      assert(
        different <= drawn * 0.01,
        `${type}/${categoryCount}: filtered resident categorical colors must equal a unfiltered resident layer whose hidden rows have null categories (${different} of ${drawn} pixels differ)`
      );
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, results}));
} catch (error) {
  console.error(error);
  console.error(
    JSON.stringify(
      await page
        .evaluate(() => ({
          native: window.__keplerComputeTest.getState().nativeLayers,
          rendered: window.__keplerComputeTest
            .renderedLayers()
            .map(layer => [layer.id, layer.visible, layer.loaded]),
          error: window.__keplerComputeTest.getState().error
        }))
        .catch(() => null)
    )
  );
  console.error(JSON.stringify(results));
  process.exitCode = 1;
} finally {
  await browser.close();
}
