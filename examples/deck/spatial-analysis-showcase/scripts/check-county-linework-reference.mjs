// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/county-linework-browser-reference.json';
const referenceFile = '/tmp/county-linework-independent-reference.json';
const referenceScript = new URL('./county-linework-reference.py', import.meta.url);
execFileSync('/tmp/ahk-showcase-venv/bin/python', [referenceScript.pathname, referenceFile], {stdio: 'inherit'});
const reference = JSON.parse(await readFile(referenceFile, 'utf8'));
const byEncoding = Object.fromEntries(reference.references.map(row => [row.encoding, row]));
const displayCases = ['shared-paths', 'noded-segments', 'polygonized-faces'];
const reports = [];
const errors = [];
let browser;
let page;
let lastBody = '';
let lastReadouts = {};

const read = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
const waitForSummaries = async encoding => {
  const expected = byEncoding[encoding];
  await page.waitForFunction(expected => {
    const body = document.body.innerText;
    const rows = Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? '']));
    return body.includes('GPU graph build failed') || body.includes('This analysis could not start') || (
      /^GPUSharedPaths (?:complete|incomplete);/.test(rows['GPU shared-path runs'] ?? '') &&
      /^GPULineNoding (?:complete|partial);/.test(rows['GPU noded-segment provenance'] ?? '') &&
      /^GPUPolygonize (?:complete|partial);/.test(rows['GPU polygonization status'] ?? '')
    );
  }, expected, {timeout: 120_000});
  const readouts = await read();
  lastReadouts = readouts;
  assert.ok(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), await page.locator('body').innerText());
  for (const label of ['GPU shared-path runs', 'GPU noded-segment provenance', 'GPU polygonization status']) {
    assert.ok(!readouts[label]?.includes('GPU graph build failed'), `${label}: ${readouts[label]}`);
    assert.ok(readouts[label], `Missing readout ${label}`);
  }
  return readouts;
};
const runCase = async (name, action) => {
  const item = {name};
  try {
    item.readouts = await action();
    lastReadouts = item.readouts;
    item.passed = true;
  } catch (error) {
    item.passed = false;
    item.error = String(error);
    item.readouts = await read().catch(() => lastReadouts);
    lastReadouts = item.readouts;
    errors.push(`${name}: ${String(error)}`);
  }
  reports.push(item);
  if (!item.passed) throw new Error(`${name}: ${item.error}`);
};
const select = (name, value) => page.locator(`#all-option-${name}`).selectOption(String(value));
const ready = async () => {
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || document.body.innerText.includes('This analysis could not start'), null, {timeout: 120_000});
  assert.ok(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), await page.locator('body').innerText());
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
};
const checkSemantics = (encoding, display, readouts) => {
  const expected = byEncoding[encoding];
  const shared = readouts['GPU shared-path runs'];
  const noding = readouts['GPU noded-segment provenance'];
  const polygonize = readouts['GPU polygonization status'];
  assert.match(shared, /GPUSharedPaths complete;/);
  assert.match(shared, /uncertain 0, overflow 0\./);
  const direction = shared.match(/forward (\d+), backward (\d+)/);
  assert.ok(direction, shared);
  if (encoding === 'reverse-right-ring') {
    assert.equal(Number(direction[2]), 0, `reversed ring has no backward runs: ${shared}`);
    assert.ok(Number(direction[1]) > 0, `reversed ring has forward shared runs: ${shared}`);
  } else {
    assert.equal(Number(direction[1]), 0, `archived boundary has no forward runs: ${shared}`);
    assert.ok(Number(direction[2]) > 0, `archived boundary has backward shared runs: ${shared}`);
  }
  const runTotals = shared.match(/(\d+) \/ (\d+) runs, (\d+) \/ (\d+) vertices/);
  assert.ok(runTotals, shared);
  assert.ok(Number(runTotals[1]) > 0 && Number(runTotals[1]) <= expected.shared.sharedSegments, shared);
  assert.equal(Number(runTotals[3]), expected.shared.sharedSegments + Number(runTotals[1]), 'shared paths retain every independently counted segment; adjacent segments may be coalesced into fewer runs');
  const perRunVertexCounts = [...shared.matchAll(/, (\d+) vertices/g)].map(match => Number(match[1]));
  if (perRunVertexCounts.length) assert.equal(perRunVertexCounts.reduce((sum, count) => sum + count - 1, 0), expected.shared.sharedSegments, 'GPU shared runs cover all independently counted shared segments');

  assert.match(noding, /GPULineNoding complete;/);
  assert.match(noding, /overflow 0, candidateOverflow 0, uncertain 0\./);
  const atomic = noding.match(/complete; (\d+) \/ (\d+) atomic segments/);
  assert.ok(atomic, noding);
  assert.equal(Number(atomic[1]), expected.rawSourceAtomicSegments, `GPULineNoding retains duplicate source segments for ${encoding}`);
  assert.equal(Number(atomic[2]), expected.rawSourceAtomicSegments, `all source segments are emitted for ${encoding}`);

  assert.match(polygonize, /GPUPolygonize complete;/);
  const faces = polygonize.match(/complete; (\d+) bounded faces/);
  assert.ok(faces, polygonize);
  assert.equal(Number(faces[1]), expected.polygonize.faces, `GEOS polygonized face count for ${encoding}`);
  const areaMatch = polygonize.match(/face areas ([\d., ]+) m²;/);
  assert.ok(areaMatch, polygonize);
  const actualAreas = areaMatch[1].split(', ').map(Number).sort((left, right) => left - right);
  assert.equal(actualAreas.length, expected.polygonize.faceAreas.length);
  for (let faceIndex = 0; faceIndex < actualAreas.length; faceIndex++) {
    assert.ok(Math.abs(actualAreas[faceIndex] - expected.polygonize.faceAreas[faceIndex]) <= 1,
      `GPU face area ${actualAreas[faceIndex]} versus GEOS ${expected.polygonize.faceAreas[faceIndex]} m² (absolute tolerance 1 m²)`);
  }
  assert.match(polygonize, /overflow 0, candidateOverflow 0, uncertain 0;/);
  assert.match(polygonize, /polygon overflow 0, candidateOverflow 0, uncertain 0;/);
  if (encoding === 'omit-outer-edge') {
    assert.match(polygonize, /Scenario omits source edge 0 from county FIPS 41015/);
    const dangles = polygonize.match(/dangles (\d+)/);
    assert.ok(dangles && Number(dangles[1]) > 0, 'open edge creates polygonizer dangle diagnostics');
  }
  if (display === 'noded-segments') assert.match(noding, /FIPS 41015|FIPS 41033/);
  return {shared, noding, polygonize};
};

try {
  browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(`pageerror: ${error.message}`));
  page.on('console', message => {if (message.type() === 'error') errors.push(`console: ${message.text()}`);});
  const url = new URL(baseUrl);
  url.hash = '#/story/county-linework';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await ready();
  await select('display', 'shared-paths');

  for (const encoding of ['archived', 'reverse-right-ring', 'omit-outer-edge']) for (const display of displayCases) {
    await runCase(`${encoding}-${display}`, async () => {
      await select('inputEncoding', encoding);
      await select('display', display);
      const readouts = await waitForSummaries(encoding);
      checkSemantics(encoding, display, readouts);
      return readouts;
    });
  }

  await runCase('cache-return-archived', async () => {
    await select('inputEncoding', 'reverse-right-ring');
    await waitForSummaries('reverse-right-ring');
    await select('inputEncoding', 'archived');
    await select('display', 'polygonized-faces');
    const readouts = await waitForSummaries('archived');
    checkSemantics('archived', 'polygonized-faces', readouts);
    return readouts;
  });
} catch (error) {
  errors.push(`harness: ${String(error)}`);
} finally {
  lastBody = page ? await page.locator('body').innerText().catch(() => '') : '';
  if (browser) await browser.close().catch(error => errors.push(`browser close: ${String(error)}`));
  await writeFile(outputFile, JSON.stringify({baseUrl, reference, reports, errors, lastBody, lastReadouts}, null, 2) + '\n');
}
assert.deepEqual(errors, [], `County linework acceptance failed; report written to ${outputFile}`);
assert.ok(reports.every(item => item.passed), `County linework cases failed; report written to ${outputFile}`);
console.log(`Accepted ${reports.length} county linework cases; report: ${outputFile}`);
