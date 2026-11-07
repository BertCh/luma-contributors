// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {captureBuildEvidence} from './earth-build-evidence.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/osm-probe-layer-refresh';
await mkdir(outputDirectory, {recursive: true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const sceneSource = await readFile(new URL('../src/scenes/dataframe/osm-history-contributors.scene.ts', import.meta.url), 'utf8');
assert.match(sceneSource, /id: 'group-moments'/, 'The expected group-moments step must exist');
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 960}});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
await page.addInitScript(() => {
  window.osmProbeGPUErrors = [];
  if (typeof GPUAdapter !== 'undefined') {
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (...arguments_) {
      const device = await requestDevice.apply(this, arguments_);
      device.addEventListener('uncapturederror', event => window.osmProbeGPUErrors.push(String(event.error)));
      return device;
    };
  }
  const markerGroups = new WeakSet();
  const boundMarkerGroups = new Set();
  const originalCreateBindGroup = GPUDevice.prototype.createBindGroup;
  GPUDevice.prototype.createBindGroup = function (descriptor) {
    const group = originalCreateBindGroup.call(this, descriptor);
    const hasProbeIndex = descriptor.entries.some(entry => {
      const resource = entry.resource;
      const buffer = resource && 'buffer' in resource ? resource.buffer : resource;
      return buffer?.label?.includes('group-z-score-probe-index');
    });
    if (hasProbeIndex) markerGroups.add(group);
    return group;
  };
  const originalBeginRenderPass = GPUCommandEncoder.prototype.beginRenderPass;
  GPUCommandEncoder.prototype.beginRenderPass = function (...arguments_) {
    boundMarkerGroups.clear();
    return originalBeginRenderPass.apply(this, arguments_);
  };
  const originalSetBindGroup = GPURenderPassEncoder.prototype.setBindGroup;
  GPURenderPassEncoder.prototype.setBindGroup = function (index, group, ...arguments_) {
    if (markerGroups.has(group)) boundMarkerGroups.add(index);
    else boundMarkerGroups.delete(index);
    return originalSetBindGroup.call(this, index, group, ...arguments_);
  };
  for (const method of ['draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect']) {
    const originalDraw = GPURenderPassEncoder.prototype[method];
    GPURenderPassEncoder.prototype[method] = function (...arguments_) {
      if (boundMarkerGroups.size > 0) window.__osmProbeLayerDraws++;
      return originalDraw.apply(this, arguments_);
    };
  }
  window.__osmProbeLayerDraws = 0;
});

const readouts = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [
  row.querySelector('dt')?.textContent ?? '',
  row.querySelector('dd')?.textContent ?? ''
])));

async function waitForProbe(value) {
  await page.waitForFunction(expected => [...document.querySelectorAll('.readout')].some(row =>
    row.querySelector('dt')?.textContent === 'GPU z-score probe' && row.querySelector('dd')?.textContent?.includes(expected)
  ), value, {timeout: 15_000});
}

async function countDrawsAcrossFrames(frameCount = 4) {
  return page.evaluate(async count => {
    const startingCount = window.__osmProbeLayerDraws;
    await new Promise(resolve => {
      let completedFrames = 0;
      const nextFrame = () => {
        completedFrames++;
        if (completedFrames >= count) resolve();
        else requestAnimationFrame(nextFrame);
      };
      requestAnimationFrame(nextFrame);
    });
    return {startingCount, frameDelta: window.__osmProbeLayerDraws - startingCount, endingCount: window.__osmProbeLayerDraws};
  }, frameCount);
}

try {
  const url = new URL(baseUrl);
  url.hash = '#/story/osm-history-contributors/group-moments';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 60_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();

  // Expand the source-year fixture independently of step defaults. Native slider stepping is retained.
  for (const [id, value] of [['years-low', 2007.5], ['years-high', 2026.4], ['inspect', 1]]) {
    await page.locator(`#all-option-${id}`).evaluate((input, next) => {
      input.value = String(next);
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    }, value);
  }

  // Contributor #1 has no sampled building nodes, but does have sampled nodes in this expanded interval.
  await page.locator('#all-option-kind').selectOption('5');
  await page.locator('#all-option-colorBy').selectOption('within-group-z-score');
  await waitForProbe('z-score unavailable');
  const noProbeInitially = await readouts();
  const absentDraws = await countDrawsAcrossFrames();
  assert.equal(absentDraws.frameDelta, 0, `probe layer drew without a selected row: ${JSON.stringify(absentDraws)}`);

  await page.locator('#all-option-kind').selectOption('all');
  await waitForProbe('GPU within-group z-score');
  const validProbe = await readouts();
  const visibleDraws = await countDrawsAcrossFrames();
  assert.match(validProbe['GPU z-score probe'], /Sample row \d+: GPU within-group z-score/);
  assert.ok(visibleDraws.frameDelta > 0, `probe layer did not draw with a selected row: ${JSON.stringify(visibleDraws)}`);

  await page.locator('#all-option-kind').selectOption('5');
  await waitForProbe('z-score unavailable');
  const noProbeFinally = await readouts();
  const finalDraws = await countDrawsAcrossFrames();
  assert.equal(finalDraws.frameDelta, 0, `probe layer still drew after its selected row disappeared: ${JSON.stringify(finalDraws)}`);
  assert.deepEqual(errors, []);
  const gpuErrors = await page.evaluate(() => window.osmProbeGPUErrors);
  assert.deepEqual(gpuErrors, [], 'Uncaptured WebGPU validation errors');

  const reportPath = `${outputDirectory}/report.json`;
  await writeFile(reportPath, JSON.stringify({passed: true, sequence: [noProbeInitially, validProbe, noProbeFinally], drawCounts: {absentDraws, visibleDraws, finalDraws}, errors, gpuErrors}, null, 2) + '\n');
  console.log(`Verified OSM probe visibility transition: ${reportPath}`);
} catch (error) {
  const gpuErrors = await page.evaluate(() => window.osmProbeGPUErrors).catch(() => []);
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: false, error: String(error), readouts: await readouts(), errors, gpuErrors}, null, 2) + '\n');
  throw error;
} finally {
  await browser.close();
  await verifyBuildEvidence();
}
