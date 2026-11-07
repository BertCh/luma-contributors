// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {scaleQuantile} from 'd3-scale';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
// Freeze loaded development modules while other agents continue editing the host.
await page.routeWebSocket('**/*', socket => {
  const server = socket.connectToServer();
  server.onMessage(message => {
    if (typeof message === 'string') {
      try {
        const packet = JSON.parse(message);
        if (packet.type === 'update' || packet.type === 'full-reload') return;
      } catch {}
    }
    socket.send(message);
  });
  socket.onMessage(message => server.send(message));
});
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (
    message.type() === 'error' &&
    /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())
  )
    failures.push(message.text());
});
const results = [];
const layerId = 'resident-points';
const palette = ['#ff0000', '#00ff00', '#0000ff', '#ffff00'];
async function waitForLegend(
  scale,
  {channel = 'fillColor', selectionAfter = -1, finiteCount} = {}
) {
  await page.waitForFunction(
    ({layerId, channel, scale, selectionAfter, finiteCount}) => {
      const api = window.__keplerComputeTest;
      const state = api.getState();
      if (state.error) throw new Error(state.error);
      const snapshot = api.getStyleLegend(layerId, channel);
      const legend = snapshot?.legend;
      return (
        legend?.scale === scale &&
        legend.revision === snapshot.expected.revision &&
        legend.source.datasetId === snapshot.expected.source.datasetId &&
        legend.source.revision === snapshot.expected.source.revision &&
        legend.selectionRevision === snapshot.expected.selectionRevision &&
        (legend.selectionRevision ?? 0) > selectionAfter &&
        (finiteCount === undefined || legend.finiteCount === finiteCount) &&
        api.renderedLayers().some(layer => layer.id === layerId && layer.loaded && layer.visible)
      );
    },
    {layerId, channel, scale, selectionAfter, finiteCount},
    {timeout: 45000}
  );
  const snapshot = await page.evaluate(
    ({layerId, channel}) => window.__keplerComputeTest.getStyleLegend(layerId, channel),
    {layerId, channel}
  );
  assert(snapshot.legend.domain.length <= 256, 'legend remains bounded metadata');
  assert(!snapshot.reason, snapshot.reason);
  return snapshot;
}
async function apply(configuration) {
  const frame = await page.evaluate(() => window.__keplerComputeTest.getState().frameCount);
  await page.evaluate(
    configuration => window.__keplerComputeTest.applyNativeConfig(configuration),
    configuration
  );
  await page.waitForFunction(
    frame => window.__keplerComputeTest.getState().frameCount > frame,
    frame,
    {timeout: 45000}
  );
}
let legendBounds = null;
function getMapPixels(image) {
  const values = [];
  for (let y = 140; y < 800; y++)
    for (let x = 440; x < 1000; x++) {
      if (
        legendBounds &&
        x >= legendBounds.x &&
        x <= legendBounds.x + legendBounds.width &&
        y >= legendBounds.y &&
        y <= legendBounds.y + legendBounds.height
      )
        continue;
      const offset = (y * image.width + x) * 4;
      values.push([image.data[offset], image.data[offset + 1], image.data[offset + 2]]);
    }
  return values;
}
function countColored(values) {
  return values.filter(
    channels => Math.max(...channels) > 120 && Math.max(...channels) - Math.min(...channels) > 60
  ).length;
}
async function screenshot(name) {
  legendBounds = await page
    .locator('.map-legend')
    .first()
    .boundingBox()
    .catch(() => null);
  const bytes = await page.screenshot({path: `/tmp/luma-kepler-style-${name}.png`});
  const image = PNG.sync.read(bytes);
  const coloredPixels = countColored(getMapPixels(image));
  assert(coloredPixels > 100, `${name} must draw map pixels, found ${coloredPixels}`);
  return {image, coloredPixels};
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=4096', {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(
    () => {
      const overlay = document.querySelector('vite-error-overlay');
      if (overlay)
        throw new Error(overlay.shadowRoot?.textContent?.slice(0, 2000) || 'Vite module error');
      const api = window.__keplerComputeTest;
      if (api?.getState().error) throw new Error(api.getState().error);
      return api?.getState().selectedCount === api?.source().length && Boolean(api?.getStyleLegend);
    },
    undefined,
    {timeout: 60000}
  );
  const original = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const initial = await page.evaluate(() => ({
    state: window.__keplerComputeTest.getState(),
    rows: window.__keplerComputeTest.source()
  }));

  // A real native field selector dispatches the same visual-channel action used by Kepler users.
  if (!(await page.locator('.layer-config-group').count()))
    await page.locator('.layer__enable-config svg').first().click({timeout: 5000});
  const colorGroup = page
    .locator('.layer-config-group')
    .filter({hasText: /Color based on/i})
    .first();
  await colorGroup.locator('.field-selector').first().click({timeout: 5000});
  await page
    .locator('.field-selector_list-item')
    .filter({has: page.getByText('value', {exact: true})})
    .click();
  await page.waitForFunction(
    () =>
      window.__keplerComputeTest.saveNativeConfig().config.visState.layers[0].visualChannels
        .colorField?.name === 'value'
  );
  await waitForLegend('quantile', {finiteCount: initial.rows.length});
  results.push({check: 'native color-field dropdown', field: 'value'});

  const numeric = structuredClone(original);
  numeric.config.uiState = {
    ...numeric.config.uiState,
    mapControls: {...numeric.config.uiState?.mapControls, mapLegend: {active: true}}
  };
  const point = numeric.config.visState.layers[0];
  point.visualChannels = {
    ...point.visualChannels,
    colorField: {name: 'value', type: 'real'},
    colorScale: 'quantile',
    sizeField: {name: 'value', type: 'real'},
    sizeScale: 'sqrt',
    strokeColorField: null
  };
  point.config.visConfig = {
    ...point.config.visConfig,
    colorRange: {name: 'Resident fixture', colors: palette},
    radiusRange: [3, 30],
    opacity: 1,
    filled: true,
    outline: false,
    fixedRadius: false
  };
  for (const filter of numeric.config.visState.filters)
    if (filter.name?.[0] === 'value') filter.fixedDomain = false;
  await apply(numeric);
  const quantile = await waitForLegend('quantile', {finiteCount: initial.rows.length});
  const values = initial.rows.map(row => row[2]);
  const native = scaleQuantile().domain(values).range(palette);
  const expectedEdges = [Math.min(...values), ...native.quantiles(), Math.max(...values)];
  quantile.legend.domain.forEach((value, index) =>
    assert(Math.abs(value - expectedEdges[index]) < 0.0001, `quantile edge ${index}`)
  );
  await page
    .locator('[data-resident-style-legend="fillColor"]')
    .waitFor({state: 'visible', timeout: 10000});
  assert(
    !/Computing|Updating/.test(
      await page.locator('[data-resident-style-legend="fillColor"]').first().innerText()
    )
  );
  // Playwright visibility alone does not detect a legend hidden under the compute sidebar.
  await page.waitForFunction(
    () => {
      const legend = document.querySelector('[data-resident-style-legend="fillColor"]');
      if (!legend) return false;
      const title = legend.querySelector('strong') ?? legend;
      const bounds = title.getBoundingClientRect();
      const hit = document.elementFromPoint(
        bounds.x + bounds.width / 2,
        bounds.y + bounds.height / 2
      );
      return Boolean(hit && legend.contains(hit));
    },
    undefined,
    {timeout: 10000}
  );
  const quantilePixels = await screenshot('quantile-sqrt');
  const quantileColors = getMapPixels(quantilePixels.image);
  const palettePixels = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0]
  ].map(
    expected =>
      quantileColors.filter(color =>
        color.every((value, index) => Math.abs(value - expected[index]) < 15)
      ).length
  );
  assert(
    palettePixels.every(count => count > 20),
    `all four quantile colors must render beyond the legend: ${palettePixels}`
  );
  results.push({
    check: 'quantile color and sqrt radius',
    coloredPixels: quantilePixels.coloredPixels,
    palettePixels,
    domain: quantile.legend.domain
  });

  // Numeric radius controls permit sqrt only. Native Line width is a size channel whose
  // actual validation allows log; exercising that layer avoids claiming every channel does.
  const logarithmic = structuredClone(numeric);
  const logarithmicLine = logarithmic.config.visState.layers[0];
  logarithmicLine.type = 'line';
  logarithmicLine.config.columnMode = 'points';
  logarithmicLine.config.columns = {
    lat0: 'latitude',
    lng0: 'longitude',
    lat1: 'targetLatitude',
    lng1: 'targetLongitude'
  };
  logarithmicLine.visualChannels.sizeScale = 'log';
  logarithmicLine.config.visConfig.sizeRange = [1, 10];
  logarithmicLine.config.visConfig.thickness = 2;
  await apply(logarithmic);
  const logSource = await waitForLegend('log', {channel: 'size', finiteCount: initial.rows.length});
  assert.equal(
    (await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig())).config.visState
      .layers[0].visualChannels.sizeScale,
    'log'
  );
  assert.equal(logSource.legend.domainStatus, 'valid');
  assert(Math.abs(logSource.legend.domain[0] - Math.min(...values)) < 0.0001);
  assert(Math.abs(logSource.legend.domain.at(-1) - Math.max(...values)) < 0.0001);
  await page.evaluate(() => window.__keplerComputeTest.setRange(40, 70));
  const logRows = initial.rows.filter(row => row[2] >= 40 && row[2] <= 70);
  const logSelected = await waitForLegend('log', {
    channel: 'size',
    selectionAfter: logSource.expected.selectionRevision,
    finiteCount: logRows.length
  });
  assert.equal(logSelected.legend.domainStatus, 'valid');
  assert(logSelected.legend.domain[0] >= 40 && logSelected.legend.domain.at(-1) <= 70);
  results.push({
    check: 'native Line logarithmic width with selected GPU domain',
    domain: logSelected.legend.domain,
    selected: logRows.length,
    revision: logSelected.expected.selectionRevision
  });

  const quantize = structuredClone(numeric);
  // Pinned Kepler's numeric color UI rejects linear and restores its quantize default.
  quantize.config.visState.layers[0].visualChannels.colorScale = 'linear';
  await apply(quantize);
  const quantizeLegend = await waitForLegend('equal-interval', {finiteCount: initial.rows.length});
  assert.equal(
    (await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig())).config.visState
      .layers[0].visualChannels.colorScale,
    'quantize'
  );
  assert(Math.abs(quantizeLegend.legend.domain[0] - Math.min(...values)) < 0.0001);
  assert(Math.abs(quantizeLegend.legend.domain.at(-1) - Math.max(...values)) < 0.0001);
  const quantizePixels = await screenshot('native-quantize');
  results.push({
    check: 'native numeric color validation (linear becomes quantize)',
    coloredPixels: quantizePixels.coloredPixels
  });

  const previousSelection = quantizeLegend.expected.selectionRevision;
  const filtered = initial.rows.filter(row => row[2] >= 40 && row[2] <= 70);
  await page.evaluate(() => window.__keplerComputeTest.setRange(40, 70));
  const selectedLegend = await waitForLegend('equal-interval', {
    selectionAfter: previousSelection,
    finiteCount: filtered.length
  });
  await page.waitForFunction(
    count => window.__keplerComputeTest.getState().selectedCount === count,
    filtered.length
  );
  assert(selectedLegend.legend.domain[0] >= 40 && selectedLegend.legend.domain.at(-1) <= 70);
  assert.equal(selectedLegend.legend.selectionRevision, selectedLegend.expected.selectionRevision);
  results.push({
    check: 'dynamic domain and legend generation',
    selected: filtered.length,
    revision: selectedLegend.expected.selectionRevision
  });

  await apply(quantize);
  const beforeAnalysis = await waitForLegend('equal-interval', {finiteCount: initial.rows.length});
  const beforeAnalysisPixels = getMapPixels((await screenshot('before-analysis')).image);
  await page.evaluate(() => window.__keplerComputeTest.analysis().setMode('z-score'));
  await page.waitForFunction(
    () => window.__keplerComputeTest.analysis().summary?.operation === 'z-score',
    undefined,
    {timeout: 45000}
  );
  const analysisPixels = getMapPixels((await screenshot('analysis-override')).image);
  const changedPixels = analysisPixels.filter((color, index) =>
    color.some((value, channel) => Math.abs(value - beforeAnalysisPixels[index][channel]) > 20)
  ).length;
  assert(changedPixels > 100, 'analysis must override rendered native fill colors');
  const duringAnalysis = await page.evaluate(
    layerId => window.__keplerComputeTest.getStyleLegend(layerId, 'fillColor'),
    layerId
  );
  assert.equal(duringAnalysis.legend.revision, beforeAnalysis.legend.revision);
  await page.evaluate(() => window.__keplerComputeTest.analysis().setMode('off'));
  results.push({check: 'analysis override precedence', changedPixels});

  const categorical = structuredClone(numeric);
  categorical.config.visState.layers[0].visualChannels.colorField = {
    name: 'category',
    type: 'integer'
  };
  categorical.config.visState.layers[0].visualChannels.colorScale = 'customOrdinal';
  categorical.config.visState.layers[0].config.visConfig.colorRange = {
    name: 'Explicit category map',
    type: 'customOrdinal',
    colors: palette,
    colorMap: palette.map((color, index) => [index, color])
  };
  await apply(categorical);
  const categories = await waitForLegend('ordinal');
  assert.deepEqual(categories.legend.labels, ['0', '1', '2', '3']);
  assert.deepEqual(categories.legend.codes, [0, 1, 2, 3]);
  const categoryPixels = await screenshot('ordinal');
  results.push({
    check: 'explicit categorical dictionary',
    labels: categories.legend.labels,
    coloredPixels: categoryPixels.coloredPixels
  });

  // Isolate one known row so a hollow native point has a background center and colored ring.
  const hollow = structuredClone(original);
  hollow.config.visState.layers[0].visualChannels = {
    ...point.visualChannels,
    colorField: null,
    sizeField: null,
    strokeColorField: null
  };
  hollow.config.visState.layers[0].config.visConfig = {
    ...point.config.visConfig,
    filled: false,
    outline: true,
    strokeColor: [255, 0, 255],
    thickness: 3,
    radius: 50,
    opacity: 1
  };
  hollow.config.mapState = {
    ...hollow.config.mapState,
    longitude: initial.rows[0][0],
    latitude: initial.rows[0][1],
    zoom: 13,
    pitch: 0,
    bearing: 0
  };
  await apply(hollow);
  await page.evaluate(
    value => window.__keplerComputeTest.setRange(value, value),
    initial.rows[0][2]
  );
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().selectedCount === 1,
    undefined,
    {timeout: 45000}
  );
  const center = await page.evaluate(() => {
    const position = window.__keplerComputeTest.projectSourceRow(0);
    const canvas = [...document.querySelectorAll('canvas')].find(canvas =>
      /deckgl/i.test(canvas.id)
    );
    const bounds = canvas?.getBoundingClientRect();
    return [position[0] + (bounds?.left ?? 0), position[1] + (bounds?.top ?? 0)];
  });
  const hollowImage = PNG.sync.read(
    await page.screenshot({path: '/tmp/luma-kepler-style-hollow.png'})
  );
  const isPink = (x, y) => {
    const offset = (y * hollowImage.width + x) * 4;
    return (
      hollowImage.data[offset] > 180 &&
      hollowImage.data[offset + 1] < 100 &&
      hollowImage.data[offset + 2] > 180
    );
  };
  const centerX = Math.round(center[0]);
  const centerY = Math.round(center[1]);
  assert(!isPink(centerX, centerY), 'hollow point center must not be filled');
  let ringPixels = 0;
  for (let y = centerY - 24; y <= centerY + 24; y++)
    for (let x = centerX - 24; x <= centerX + 24; x++) if (isPink(x, y)) ringPixels++;
  assert(ringPixels > 12, `hollow native point must draw its outline; found ${ringPixels}`);
  results.push({check: 'native hollow outline', center: [centerX, centerY], ringPixels});

  await apply(original);
  await page.waitForFunction(
    count => window.__keplerComputeTest.getState().selectedCount === count,
    initial.rows.length
  );
  assert.equal(
    (await page.evaluate(() => window.__keplerComputeTest.getState())).uploadBytes,
    initial.state.uploadBytes,
    'style/filter changes must not reupload source vectors'
  );
  assert.deepEqual(failures, []);
  process.stdout.write(
    `${JSON.stringify({passed: true, sourceUploadBytes: initial.state.uploadBytes, results}, null, 2)}\n`
  );
} catch (error) {
  console.error(
    JSON.stringify(
      await page.evaluate(() => ({
        state: window.__keplerComputeTest?.getState(),
        legend: window.__keplerComputeTest?.getStyleLegend?.('resident-points', 'fillColor'),
        config: window.__keplerComputeTest?.saveNativeConfig(),
        moduleError: document
          .querySelector('vite-error-overlay')
          ?.shadowRoot?.textContent?.slice(0, 2000)
      })),
      null,
      2
    )
  );
  console.error((await page.locator('body').innerText()).slice(0, 12000));
  await page.screenshot({path: '/tmp/luma-kepler-style-failure.png'});
  throw error;
} finally {
  await browser.close();
}
