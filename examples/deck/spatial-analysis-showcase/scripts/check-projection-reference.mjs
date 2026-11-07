// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/projection-reference-browser.json';
const reference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/projection-contract-reference.json', import.meta.url), 'utf8'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 960}});
const reports = [];
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
const readouts = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
async function choose(id, name) {
  await page.locator(`[role="radiogroup"][aria-labelledby="all-option-${id}-label"]`).getByRole('radio', {name, exact: true}).click();
}
function close(actual, expected, context) {
  assert.ok(Math.abs(actual - expected) <= 0.02, `${context}: ${actual} vs ${expected}`);
}
try {
  const url = new URL(baseUrl);
  url.hash = '#/story/sinusoidal-projection';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready && globalThis.spatialAnalysisShowcase?.sceneId === 'sinusoidal-projection', null, {timeout: 60_000});
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const [hub, label] of [['ORD', "Chicago O'Hare (ORD)"], ['HND', 'Tokyo Haneda (HND)'], ['SYD', 'Sydney (SYD)']]) {
    await choose('hub', label);
    // Published GPUSinusoidalProjection has no wrapLongitude option; only the raw cases remain.
    for (const wrap of [false]) {
      for (const axis of ['x', 'y']) {
        await choose('axis', axis === 'x' ? 'x · west/east' : 'y · south/north');
        await page.waitForFunction(({hub, axis}) => {
          const rows = Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? '']));
          return rows['Origin and LAX coordinates']?.startsWith(`${hub} (x, y km)`) && rows['GPU coordinate support']?.endsWith(`km on ${axis}`);
        }, {hub, axis}, {timeout: 30_000});
        const observed = await readouts();
        const expected = reference.cases.find(value => value.hub === hub && value.wrapLongitude === wrap);
        const support = observed['GPU coordinate support'].match(/^([\d,]+) finite GPU rows · ([-\d.]+) to ([-\d.]+) km/);
        assert.ok(support, observed['GPU coordinate support']);
        assert.equal(Number(support[1].replaceAll(',', '')), 3257);
        close(Number(support[2]), expected[axis === 'x' ? 'minimumX' : 'minimumY'], 'minimum');
        close(Number(support[3]), expected[axis === 'x' ? 'maximumX' : 'maximumY'], 'maximum');
        const probe = observed['Origin and LAX coordinates'].match(/LAX \(x, y km\): ([-\d.]+), ([-\d.]+)/);
        assert.ok(probe, observed['Origin and LAX coordinates']);
        const raw = reference.cases.find(value => value.hub === hub && !value.wrapLongitude).laxProjection;
        for (const [index, value] of raw.entries()) close(Number(probe[index + 1]), value, 'LAX probe');
        assert.ok(observed['Origin and LAX coordinates'].includes('(x, y km): 0.000, 0.000'));
        reports.push({hub, wrap, axis, observed});
      }
    }
  }
  assert.deepEqual(errors, []);
  await page.screenshot({path: outputFile.replace(/\.json$/, '-verified.png')});
} catch (error) {
  reports.push({failure: String(error), readouts: await readouts(), visibleText: await page.locator('body').innerText()});
  await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true});
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, reports, errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} projection scenarios: ${outputFile}`);
