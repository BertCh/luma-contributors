// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check that Jenks thresholds equal the pinned native naturalBreaks at every size.
// Usage: node check-jenks-gpu.mjs [rows=N]. Defaults to rows=300000 (selected values read back
// in chunks and run in a Web Worker, plain and with a range-filter mask), then rows=4096.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {dirname} from 'node:path';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const require = createRequire(import.meta.url);
// The package index pulls browser-only modules, so load the Jenks module directly.
const naturalBreaks = require(`${dirname(require.resolve('@kepler.gl/utils'))}/jenks.js`).default;
const layerId = 'resident-points';
const palette = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff'];
const errorPattern =
  /WGSL|Invalid.*(?:Pipeline|Buffer|BindGroup)|validation error|exceeds the maximum|Uncaught|destroyed|readback cap|row count/i;

async function runCase(rows) {
  const browser = await chromium.launch(
    getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
  );
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  const watchdog = setTimeout(() => {
    console.error(`check-jenks-gpu rows=${rows} exceeded 140 seconds`);
    void browser.close();
  }, 140000);
  await pinViteModules(page);
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error' && errorPattern.test(message.text()))
      errors.push(message.text());
  });
  try {
    await page.goto(`http://127.0.0.1:5318/?rows=${rows}`, {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(
      () => {
        const api = window.__keplerComputeTest;
        if (api?.getState().error) throw new Error(api.getState().error);
        return api?.getState().selectedCount === api?.source().length && Boolean(api?.getStyleLegend);
      },
      undefined,
      {timeout: 90000}
    );
    const original = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    const configuration = structuredClone(original);
    const point = configuration.config.visState.layers[0];
    point.visualChannels = {
      ...point.visualChannels,
      colorField: {name: 'value', type: 'real'},
      colorScale: 'jenks',
      strokeColorField: null
    };
    point.config.visConfig = {
      ...point.config.visConfig,
      colorRange: {name: 'Jenks fixture', colors: palette},
      opacity: 1,
      filled: true
    };
    for (const filter of configuration.config.visState.filters)
      if (filter.name?.[0] === 'value') filter.fixedDomain = false;
    await page.evaluate(
      configuration => window.__keplerComputeTest.applyNativeConfig(configuration),
      configuration
    );
    await page.waitForFunction(
      ({layerId}) => {
        const api = window.__keplerComputeTest;
        const state = api.getState();
        if (state.error) throw new Error(state.error);
        const snapshot = api.getStyleLegend(layerId, 'fillColor');
        return (
          snapshot?.legend?.scale === 'threshold' &&
          snapshot.legend.revision === snapshot.expected.revision &&
          api.renderedLayers().some(layer => layer.id === layerId && layer.loaded && layer.visible)
        );
      },
      {layerId},
      {timeout: 90000}
    ).catch(async error => {
      console.error(JSON.stringify(await page.evaluate(({layerId}) => ({
        state: window.__keplerComputeTest.getState().error,
        legend: window.__keplerComputeTest.getStyleLegend(layerId, 'fillColor'),
        rendered: window.__keplerComputeTest.renderedLayers()
      }), {layerId})), errors);
      throw error;
    });
    const snapshot = await page.evaluate(
      ({layerId}) => window.__keplerComputeTest.getStyleLegend(layerId, 'fillColor'),
      {layerId}
    );
    const state = await page.evaluate(() => window.__keplerComputeTest.getState());
    const sourceRows = await page.evaluate(() => window.__keplerComputeTest.source().map(row => row[2]));
    assert(!snapshot.reason, snapshot.reason);
    assert(!state.error, state.error);
    const legend = snapshot.legend;
    console.log(JSON.stringify({rows, legendDomain: legend.domain, finiteCount: legend.finiteCount}));
    const finite = sourceRows.filter(Number.isFinite);
    const interior = legend.domain;
    assert.equal(interior.length, palette.length - 1, 'one threshold per class boundary');
    // The thresholds are exactly native naturalBreaks over the Float32 values, at any row count.
    const expected = naturalBreaks(Array.from(new Float32Array(finite)), palette.length);
    console.log(JSON.stringify({expected, interior}));
    assert.deepEqual(interior, expected, 'thresholds are exactly native naturalBreaks');
    if (rows > 262144) {
      // A range filter makes the selected-domain mask part of the readback.
      const before = snapshot.expected.selectionRevision;
      await page.evaluate(() => window.__keplerComputeTest.setRange(30, 70));
      await page.waitForFunction(
        ({layerId, before}) => {
          const api = window.__keplerComputeTest;
          if (api.getState().error) throw new Error(api.getState().error);
          const current = api.getStyleLegend(layerId, 'fillColor');
          return (
            current?.legend?.scale === 'threshold' &&
            current.legend.selectionRevision === current.expected.selectionRevision &&
            current.expected.selectionRevision > before
          );
        },
        {layerId, before},
        {timeout: 90000}
      );
      const selected = await page.evaluate(
        ({layerId}) => window.__keplerComputeTest.getStyleLegend(layerId, 'fillColor'),
        {layerId}
      );
      const selectedValues = (await page.evaluate(() => window.__keplerComputeTest.source())).filter(
        row => row[2] >= 30 && row[2] <= 70
      ).map(row => row[2]);
      const selectedExpected = naturalBreaks(Array.from(new Float32Array(selectedValues)), palette.length);
      console.log(JSON.stringify({selectedValues: selectedValues.length, domain: selected.legend.domain, selectedExpected}));
      assert(selectedValues.length > 0 && selectedValues.length < finite.length);
      assert.deepEqual(selected.legend.domain, selectedExpected, 'masked thresholds are exactly native naturalBreaks');
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({pass: true, rows}));
  } finally {
    clearTimeout(watchdog);
    await browser.close();
  }
}

const rows = Number(new URLSearchParams(process.argv[2] ?? '').get('rows') ?? 300000);
await runCase(rows);
await runCase(4096);
