// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/earthquake-pair-browser-reference.json';
const referenceFile = outputFile.replace(/\.json$/, '-cpu.json');
execFileSync(process.execPath, [fileURLToPath(new URL('./earthquake-pair-reference.mjs', import.meta.url)), referenceFile], {stdio: 'ignore'});
const reference = JSON.parse(await readFile(referenceFile, 'utf8'));
const reports = [];
const errors = [];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
let page;
try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  const url = new URL(baseUrl);
  url.hash = '#/story/earthquake-pair-structure';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || document.body.innerText.includes('This analysis could not start'), null, {timeout: 120000});
  assert.ok(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), await page.locator('body').innerText());
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  const read = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
  const wait = label => page.waitForFunction(name => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent === name && row.querySelector('dd')?.textContent?.startsWith('At 80 km:')), label, {timeout: 120000});
  const close = (actual, expected, name) => assert.ok(Math.abs(actual - expected) <= Math.max(0.03, Math.abs(expected) * 0.00005), `${name}: ${actual} versus ${expected}`);
  for (const scenario of reference.scenarios) {
    const checkbox = page.locator('#all-option-excludeExactTen');
    if (await checkbox.isChecked() !== scenario.excludeExactTen) await checkbox.locator('..').click();
    await page.locator('#all-option-markedCorrection').selectOption(scenario.correction);
    await page.locator('#all-option-depthPair').selectOption(scenario.pair);
    await wait('GPU pair values at 80 km');
    const readouts = await read();
    const match = readouts['GPU pair values at 80 km'].replaceAll(',', '').match(/K=([\deE+.\-]+) m²; L−r=([\deE+.\-]+) m; g=([\deE+.\-]+); mark correlation=([\deE+.\-]+); (\d+) directed pairs/);
    assert.ok(match, JSON.stringify(readouts));
    const expected = scenario.bins.at(-1);
    close(Number(match[1]), expected.k, 'K');
    close(Number(match[2]), expected.lMinusR, 'L-minus-r');
    close(Number(match[3]), expected.pairCorrelation, 'pair correlation');
    close(Number(match[4]), expected.markCorrelation, 'mark correlation');
    assert.equal(Number(match[5]), expected.cumulativeCount);
    assert.match(readouts['Marked-pair GPU status'], /GPU bins valid/);
    reports.push({kind: 'marked', scenario: {pair: scenario.pair, correction: scenario.correction, excludeExactTen: scenario.excludeExactTen}, expected, readouts});
    console.log(`Accepted ${scenario.pair}, ${scenario.correction}, exclude exact ten ${scenario.excludeExactTen}`);
  }
  await page.locator('#all-option-mode').selectOption('inhomogeneous');
  await page.locator('#all-option-inhomogeneousCorrection').selectOption('translation');
  await wait('GPU inhomogeneous values at 80 km');
  const readouts = await read();
  const match = readouts['GPU inhomogeneous values at 80 km'].match(/K=([\deE+.\-]+) m²; L−r=([\deE+.\-]+) m; g=([\deE+.\-]+)\./);
  assert.ok(match, JSON.stringify(readouts));
  const expected = reference.scenarios.find(item => !item.excludeExactTen && item.correction === 'translation' && item.pair === 'all').bins.at(-1);
  close(Number(match[1]), expected.k, 'constant intensity K');
  close(Number(match[2]), expected.lMinusR, 'constant intensity L-minus-r');
  close(Number(match[3]), expected.pairCorrelation, 'constant intensity pair correlation');
  assert.match(readouts['Inhomogeneous GPU status'], /GPU statuses valid at all 16 radii/);
  reports.push({kind: 'constant-intensity-translation', expected, readouts});
  assert.deepEqual(errors, []);
} finally {
  const lastBody = page ? await page.locator('body').innerText().catch(() => '') : '';
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, referenceFile, reports, errors, lastBody}, null, 2) + '\n');
}
