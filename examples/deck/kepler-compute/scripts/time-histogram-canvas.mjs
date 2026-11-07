// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Loads a >4000-bin hourly time dataset and compares chart render time for SVG versus canvas bars.
// Usage: node scripts/time-histogram-canvas.mjs [url]  (default headless; not part of any gate)
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const HOURS = 6000;
const START = Date.parse('2024-01-01T00:00:00Z');
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1600, height: 1100}});
page.setDefaultTimeout(60000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => {
  errors.push(error.message);
  console.error(error.stack);
});
const card = page.locator('.resident-chart-card').first();
const plot = card.getByRole('status', {name: 'Chart category summary', exact: true});

async function measureRender(label, readySelector) {
  // Toggle the interval away and back so the chart re-renders under the current renderer flag.
  await card.getByLabel('Calendar interval (UTC)', {exact: true}).selectOption('day');
  // Day bins (about 250) fall under the canvas threshold: no canvas, and few SVG rects.
  await page.waitForFunction(() => {
    const root = document.querySelector('[aria-label="Chart category summary"]');
    return Boolean(root) && !root.querySelector('canvas') && root.querySelectorAll('rect').length < 1000;
  });
  const start = Date.now();
  await card.getByLabel('Calendar interval (UTC)', {exact: true}).selectOption('hour');
  const startMark = await page.evaluate(() => performance.now());
  // Native Kepler bars cap interactive DOM bars at 1,000, far above the day-bin count.
  if (readySelector === 'svg') {
    await page.waitForFunction(
      () =>
        document.querySelectorAll('[aria-label="Chart category summary"] .bar-chart [role=button]')
          .length >= 1000
    );
  } else await page.waitForSelector(readySelector);
  const elapsed = await page.evaluate(
    mark => performance.now() - mark,
    startMark
  );
  const nodeCount = await plot.evaluate(element => element.querySelectorAll('*').length);
  console.log(
    `${label}: ready in ${elapsed.toFixed(0)} ms (wall ${Date.now() - start} ms), plot DOM nodes ${nodeCount}`
  );
  return {elapsed, nodeCount};
}

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const source = new arrow.Table({
    time: arrow.vectorFromArray(
      Array.from({length: HOURS}, (_, index) => START + index * 3600000),
      new arrow.TimestampMillisecond('UTC')
    )
  });
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'time-histogram-canvas.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    rows => window.__keplerComputeTest.getState().publications[0]?.selectedCount === rows,
    HOURS
  );
  const publication = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  await closeNativeDialog(page);
  await page.getByRole('button', {name: 'Toggle charts', exact: true}).click();
  // Kepler creates a chart card only after a chart type is chosen.
  await page.locator('.chart-config__type').click();
  await page.getByText('Bar', {exact: true}).click();
  await card.getByLabel('Chart dataset', {exact: true}).selectOption(publication.id);
  await card.getByLabel('Chart categories', {exact: true}).selectOption('time');
  await card.getByLabel('Chart binning', {exact: true}).selectOption('timeBin');
  // Native bars treat numGroups as an equal-width bin count; zero selects calendar periods.
  await card.getByLabel('Maximum displayed groups', {exact: true}).fill('0');
  await card.getByLabel('Calendar interval (UTC)', {exact: true}).selectOption('hour');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('count');

  await page.waitForSelector('[aria-label="Chart category summary"] canvas');
  const canvas = await measureRender('canvas', '[aria-label="Chart category summary"] canvas');
  // Hover tooltip and DPR backing store.
  await plot.locator('canvas').scrollIntoViewIfNeeded();
  const box = await plot.locator('canvas').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForSelector('[role=tooltip]');
  const backing = await plot.locator('canvas').evaluate(element => ({
    width: element.width,
    cssWidth: element.getBoundingClientRect().width,
    ratio: window.devicePixelRatio
  }));
  assert(Math.abs(backing.width - Math.round(backing.cssWidth * backing.ratio)) <= 2, 'Canvas backing store must follow DPR');

  await page.evaluate(() => {
    window.__residentChartForceSvgBars = true;
  });
  const svg = await measureRender('svg', 'svg');
  assert(canvas.nodeCount < svg.nodeCount, 'Canvas must render far fewer DOM nodes than SVG');
  console.log(
    `bins=${HOURS} canvas=${canvas.elapsed.toFixed(0)}ms svg=${svg.elapsed.toFixed(0)}ms nodes ${canvas.nodeCount} vs ${svg.nodeCount}`
  );
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
}
