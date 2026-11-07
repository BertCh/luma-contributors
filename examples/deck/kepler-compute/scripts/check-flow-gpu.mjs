// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check for the GPU-resident Flow aggregation: compares the WebGPU location dedupe and
// origin-destination aggregation against the CPU reference on the same tables, including a large
// multi-batch table with a selection mask. Requires a WebGPU browser and a running dev server.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const ERROR_PATTERN = /WGSL|Invalid.*(?:Pipeline|Buffer|BindGroup)|validation error|exceeds the maximum|Uncaught|destroyed/i;

function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function makeTable(rowCount, locationCount, seed) {
  const random = makeRandom(seed);
  const locations = Array.from({length: locationCount}, () => [
    Math.fround(-180 + random() * 360),
    Math.fround(-80 + random() * 160)
  ]);
  const columns = {sourceLongitude: [], sourceLatitude: [], targetLongitude: [], targetLatitude: [], count: [], selected: []};
  for (let row = 0; row < rowCount; row++) {
    const source = locations[Math.floor(random() * locationCount)];
    const target = locations[Math.floor(random() * locationCount)];
    columns.sourceLongitude.push(source[0]);
    columns.sourceLatitude.push(source[1]);
    columns.targetLongitude.push(target[0]);
    columns.targetLatitude.push(target[1]);
    columns.count.push(Math.floor(random() * 8) + 1);
    columns.selected.push(random() < 0.6 ? 1 : 0);
  }
  return columns;
}

function makeEdgeTable() {
  const columns = makeTable(40, 5, 7);
  // Non-finite endpoints, negative zero, null-like counts and duplicate rows.
  columns.sourceLongitude[3] = Number.NaN;
  columns.targetLatitude[5] = Number.POSITIVE_INFINITY;
  columns.sourceLongitude[6] = -0;
  columns.sourceLatitude[6] = 0;
  columns.targetLongitude[6] = 0;
  columns.targetLatitude[6] = -0;
  columns.count[8] = Number.NaN;
  return columns;
}

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1280, height: 900}});
const errors = [];
const watchdog = setTimeout(() => {
  console.error('Flow GPU check exceeded 140 seconds');
  void browser.close();
  process.exit(1);
}, 140000);
page.on('pageerror', error => errors.push(`pageerror: ${error.message}`));
page.on('console', message => {
  if (message.type() === 'error' && ERROR_PATTERN.test(message.text())) errors.push(message.text());
});

function compareSummary(label, gpu, reference) {
  assert.equal(gpu.pairCount, reference.pairCount, `${label}: pair count`);
  assert.deepEqual(gpu.sourceIds, reference.sourceIds, `${label}: source ids`);
  assert.deepEqual(gpu.targetIds, reference.targetIds, `${label}: target ids`);
  assert.deepEqual(gpu.firstRows, reference.firstRows, `${label}: first rows`);
  gpu.counts.forEach((count, index) =>
    assert.ok(Math.abs(count - reference.counts[index]) <= Math.max(1e-3, Math.abs(reference.counts[index]) * 1e-5), `${label}: count ${index} ${count} vs ${reference.counts[index]}`)
  );
}

async function check(label, columns) {
  const result = await page.evaluate(input => window.__keplerComputeTest.runFlowGpuCheck(input), columns);
  assert.deepEqual(result.gpuFirstEndpoints, result.referenceFirstEndpoints, `${label}: first endpoints`);
  assert.deepEqual(result.gpuBounds, result.referenceBounds, `${label}: bounds`);
  compareSummary(`${label} unfiltered`, result.gpuUnfiltered, result.referenceUnfiltered);
  compareSummary(`${label} selected`, result.gpuSelected, result.referenceSelected);
  compareSummary(`${label} selected repeat`, result.gpuSelectedRepeat, result.referenceSelected);
  console.log(JSON.stringify({
    label,
    rows: result.rowCount,
    locations: result.gpuFirstEndpoints.length,
    unfilteredPairs: result.gpuUnfiltered.pairCount,
    selectedPairs: result.gpuSelected.pairCount
  }));
}

try {
  await pinViteModules(page);
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await check('small', makeTable(200, 20, 1));
  await check('edge cases', makeEdgeTable());
  await check('multi-batch', {...makeTable(5000, 300, 2), batchCount: 7});
  await check('large', {...makeTable(1_000_000, 20000, 3), batchCount: 4});
  assert.deepEqual(errors, [], `WebGPU or page errors:\n${errors.join('\n')}`);
  console.log('Flow GPU check passed');
} catch (error) {
  console.error(error);
  if (errors.length) console.error('Collected errors:\n' + errors.join('\n'));
  process.exitCode = 1;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
