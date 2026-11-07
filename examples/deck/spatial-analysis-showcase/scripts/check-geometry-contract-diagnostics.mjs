// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Bounded diagnostic capture, not an independent numerical acceptance test.
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const directory = path.resolve(process.argv[3] ?? '/tmp/geometry-contract-diagnostics');
await mkdir(directory, {recursive: true});
const getEvidence = async () => (await fetch(new URL('build-evidence.json', baseUrl), {headers: {connection: 'close'}})).json();
const evidence = await getEvidence();
await writeFile(path.join(directory, 'build-evidence.json'), JSON.stringify(evidence, null, 2));
const results = [];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
try {
  for (const [scene, step, readout, terminal, triangulationMethod] of [
    ['sampling-frames', 'area-frame', 'Design status', 'GPU triangle frame:'],
    ['footprint-shape-lab', 'offset', 'Geometric evidence', 'Raw noding diagnostic:']
  ]) {
    const page = await browser.newPage({viewport: {width: 1440, height: 960}});
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
    const url = new URL(baseUrl);
    url.searchParams.set('geometryDiagnostics', '1');
    if (triangulationMethod) url.searchParams.set('geometryTriangulationMethod', triangulationMethod);
    url.hash = `#/story/${scene}/${step}`;
    try {
      await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
      if (scene === 'sampling-frames') {
        await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 120000});
        await page.getByRole('tab', {name: 'All controls', exact: false}).click();
        const legacy = page.locator('#all-option-includeLegacy');
        if (await legacy.isChecked()) {
          const rebuilds = await page.evaluate(() => globalThis.spatialAnalysisShowcase.rebuildCount);
          await legacy.locator('..').click();
          await page.waitForFunction(previous => globalThis.spatialAnalysisShowcase.rebuildCount > previous, rebuilds, {timeout: 120000});
        }
      }
      await page.waitForFunction(({readout, terminal}) => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent?.trim() === readout && row.querySelector('dd')?.textContent?.includes(terminal)) || document.body.innerText.includes('This analysis could not start'), {readout, terminal}, {timeout: 120000});
      const readouts = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent?.trim(), row.querySelector('dd')?.textContent?.trim()])));
      results.push({scene, step, triangulationMethod: triangulationMethod ?? 'default', readouts, errors});
      await page.screenshot({path: path.join(directory, `${scene}-${triangulationMethod ?? 'default'}.png`)});
    } catch (error) {
      results.push({scene, step, error: String(error), errors, body: await page.locator('body').innerText()});
    } finally {
      await page.close();
    }
  }
} finally {
  await browser.close();
  const after = await getEvidence();
  await writeFile(path.join(directory, 'build-evidence-after.json'), JSON.stringify(after, null, 2));
  await writeFile(path.join(directory, 'report.json'), JSON.stringify({bundleSha256: evidence.bundleSha256, bundleUnchanged: evidence.bundleSha256 === after.bundleSha256, diagnosticOnly: true, results}, null, 2));
  assert.equal(after.bundleSha256, evidence.bundleSha256);
}
console.log(JSON.stringify(results, null, 2));
