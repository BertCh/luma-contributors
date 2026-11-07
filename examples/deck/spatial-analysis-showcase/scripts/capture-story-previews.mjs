// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Capture actual rendered maps, retaining their legends and source attribution.
// Run against a stable production preview rather than a changing Vite dev server.
import {mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const SHOWCASE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_SCENES = ['transit-frequency', 'burn-severity', 'h3-aggregation', 'watersheds', 'rainfall-interpolation'];
const argumentsByName = Object.fromEntries(process.argv.slice(2).map(argument => {
  const separator = argument.indexOf('=');
  return separator < 0 ? [argument, true] : [argument.slice(0, separator), argument.slice(separator + 1)];
}));
if (argumentsByName['--help']) {
  console.log('node scripts/capture-story-previews.mjs [--url=http://127.0.0.1:5313/] [--scenes=id,id] [--output=directory]');
  process.exit(0);
}
const baseUrl = argumentsByName['--url'] ?? 'http://127.0.0.1:5313/';
const buildEvidence = await (await fetch(new URL('build-evidence.json', baseUrl))).json();
const sceneIdentifiers = argumentsByName['--scenes']?.split(',') ?? DEFAULT_SCENES;
const outputDirectory = path.resolve(argumentsByName['--output'] ?? path.join(SHOWCASE_ROOT, 'public/images/story-previews'));
await mkdir(outputDirectory, {recursive: true});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const captures = [];
try {
  for (const sceneIdentifier of sceneIdentifiers) {
    if (!/^[a-z0-9-]+$/.test(sceneIdentifier)) throw new Error(`Invalid scene id: ${sceneIdentifier}`);
    const page = await browser.newPage({viewport: {width: 1150, height: 580}, deviceScaleFactor: 1});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    try {
      const url = new URL(baseUrl);
      url.hash = `#/story/${sceneIdentifier}`;
      await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
      await page.waitForFunction(identifier => {
        const showcase = globalThis.spatialAnalysisShowcase;
        return showcase?.sceneId === identifier && (showcase.ready || showcase.error);
      }, sceneIdentifier, {timeout: 120_000});
      const state = await page.evaluate(() => ({ready: globalThis.spatialAnalysisShowcase.ready, error: globalThis.spatialAnalysisShowcase.error}));
      if (state.error || !state.ready) throw new Error(state.error ?? 'Map did not become ready.');
      // Wait for map tiles and initial asynchronous contributor summaries to settle.
      await page.waitForTimeout(2500);
      await page.waitForFunction(() => {
        const values = document.querySelectorAll(
          '.step-detail:not([hidden]) [data-readout-value], .step-detail:not([hidden]) .live-readout, .status-line'
        );
        return [...values].every(element =>
          !/^(?:rebuilding|recomputing|computing|compiling|loading|solving)\b/i.test(element.textContent.trim())
        );
      }, null, {timeout: 120_000});
      if (errors.length) throw new Error(errors.join('\n'));
      const map = page.locator('[aria-label="Map"]').first();
      const filename = `${sceneIdentifier}.png`;
      await map.screenshot({path: path.join(outputDirectory, filename)});
      const bounds = await map.boundingBox();
      captures.push({sceneId: sceneIdentifier, stepIndex: 0, filename, width: bounds.width, height: bounds.height});
      console.log(`Captured ${sceneIdentifier}`);
    } finally {
      await page.close();
    }
  }
} finally {
  await browser.close();
}
const finalBuildEvidence = await (await fetch(new URL('build-evidence.json', baseUrl))).json();
if (finalBuildEvidence.bundleSha256 !== buildEvidence.bundleSha256) {
  throw new Error('The served build changed during preview capture; rerun against one build.');
}
await writeFile(path.join(outputDirectory, 'captures.json'), `${JSON.stringify({
  description: 'Unmodified WebGPU map screenshots, including legends and dataset/basemap attribution. Illustrative previews; consult each live story for controls, evidence and limitations.',
  capturedAt: new Date().toISOString(),
  bundleSha256: buildEvidence.bundleSha256,
  captures
}, null, 2)}\n`);
