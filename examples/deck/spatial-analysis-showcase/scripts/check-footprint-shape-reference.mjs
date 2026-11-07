// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/footprint-shape-browser-reference.json';
const referenceFile = '/tmp/footprint-shape-independent-reference.json';
const referenceScript = new URL('./footprint-shape-reference.mjs', import.meta.url);
execFileSync(process.execPath, [referenceScript.pathname, '/tmp/ahk-showcase-venv/bin/python', referenceFile, '--reference-only'], {stdio: 'inherit'});
const reference = JSON.parse(await readFile(referenceFile, 'utf8'));
const byId = Object.fromEntries(reference.references.map(row => [row.sourceId === 354712 ? 'courtyard' : 'convex', row]));
const reports = [];
const errors = [];
let browser;
let page;
let lastBody = '';
let lastReadouts = {};

const read = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
const waitFor = async (label, matcher) => {
  await page.waitForFunction(([name, expression]) => {
    const body = document.body.innerText;
    const values = [...document.querySelectorAll('.readout')].filter(row => row.querySelector('dt')?.textContent === name).map(row => row.querySelector('dd')?.textContent ?? '');
    const statusRow = [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === 'GPU result');
    const status = statusRow?.querySelector('dd')?.textContent ?? '';
    return body.includes('GPU graph could not compile') || body.includes('GPU graph build failed') || (status.length > 0 && !/Compiling|pending|—/.test(status)) || values.some(value => new RegExp(expression).test(value));
  }, [label, matcher.source], {timeout: 120_000});
  const readouts = await read();
  lastReadouts = readouts;
  const body = await page.locator('body').innerText();
  assert.ok(!body.includes('GPU graph could not compile') && !body.includes('GPU graph build failed'), body);
  const value = readouts[label] ?? '';
  assert.ok(!value.includes('GPU graph could not compile') && !value.includes('graph build failed'), `${label}: ${value}`);
  assert.match(value, matcher, `${label}: ${value}`);
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
const setSlider = async (name, value) => page.locator(`#all-option-${name}`).evaluate((element, next) => {
  element.value = String(next);
  element.dispatchEvent(new Event('input', {bubbles: true}));
  element.dispatchEvent(new Event('change', {bubbles: true}));
}, value);
const ready = async () => {
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || document.body.innerText.includes('This analysis could not start'), null, {timeout: 120_000});
  assert.ok(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), await page.locator('body').innerText());
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
};
const setStudy = async (footprint, question) => {
  await select('footprint', footprint);
  await select('question', question);
};

try {
  browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(`pageerror: ${error.message}`));
  page.on('console', message => {if (message.type() === 'error') errors.push(`console: ${message.text()}`);});
  const url = new URL(baseUrl);
  url.hash = '#/story/footprint-shape-lab';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await ready();

  for (const footprint of ['courtyard', 'convex']) {
    for (const tolerance of [0.05, 0.25, 1]) {
      await runCase(`circle-${footprint}-resolved-${tolerance}`, async () => {
        await setStudy(footprint, 'circle');
        await select('budget', 'resolved');
        await setSlider('tolerance', tolerance);
        const values = await waitFor('Geometric evidence', /Radius [\d.]+ m/);
        const match = values['Geometric evidence'].match(/Radius ([\d.]+) m · remaining geometric search error ([\d.]+) m/);
        assert.ok(match, values['Geometric evidence']);
        const radius = Number(match[1]);
        const remainingError = Number(match[2]);
        assert.ok(Math.abs(radius - byId[footprint].circleRadius) <= tolerance + 0.15, `${footprint} radius ${radius}; GEOS ${byId[footprint].circleRadius}; tolerance ${tolerance} plus float32 margin`);
        assert.ok(remainingError <= tolerance + 0.01, `${footprint} search error ${remainingError} <= ${tolerance}`);
        assert.match(values['GPU result'], /Search tolerance met/);
        return values;
      });
    }
    await runCase(`circle-${footprint}-quick-bounded`, async () => {
      await setStudy(footprint, 'circle');
      await select('budget', 'quick');
      await setSlider('tolerance', 0.25);
      const values = await waitFor('Geometric evidence', /Radius [\d.]+ m/);
      const match = values['Geometric evidence'].match(/Radius ([\d.]+) m · remaining geometric search error ([\d.]+) m/);
      assert.ok(match, values['Geometric evidence']);
      assert.ok(Number(match[1]) >= 0 && Number(match[2]) >= 0, 'quick mode returns a bounded numeric result');
      assert.match(values['GPU result'], /Budget exhausted · bounded answer shown|Search tolerance met/);
      assert.match(values['Geometric evidence'], /reference acceptance withheld because the search budget was exhausted|within tolerance/);
      return values;
    });
    await runCase(`minimum-clearance-${footprint}`, async () => {
      await setStudy(footprint, 'clearance');
      const values = await waitFor('Geometric evidence', /Fragile stored vertex/);
      const match = values['GPU result'].match(/Minimum displacement ([\d.]+) m/);
      assert.ok(match, values['GPU result']);
      const actual = Number(match[1]);
      assert.ok(Math.abs(actual - byId[footprint].minimumClearance) <= 0.02, `${footprint} clearance ${actual}; GEOS ${byId[footprint].minimumClearance} m ± 0.02 m`);
      return values;
    });
  }

  for (const direction of ['outward', 'inward']) for (const join of ['round', 'mitre', 'bevel']) {
    await runCase(`convex-buffer-${direction}-${join}`, async () => {
      await setStudy('convex', 'move');
      await select('bufferDirection', direction);
      await select('join', join);
      await setSlider('offsetDistance', 4);
      const values = await waitFor('Geometric evidence', /Topology-resolved offset surface area [\d.]+ m²/);
      const match = values['Geometric evidence'].match(/Topology-resolved offset surface area ([\d.]+) m²/);
      assert.ok(match, values['Geometric evidence']);
      const actual = Number(match[1]);
      const expected = byId.convex.buffers.find(item => item.distance === (direction === 'outward' ? 4 : -4) && item.join === join);
      assert.ok(expected?.valid && !expected.empty);
      assert.ok(Math.abs(actual - expected.area) <= 2, `${direction} ${join} area ${actual}; GEOS ${expected.area} m² ± 2 m²`);
      assert.match(values['GPU result'], /GPU buffer polygon\(s\)/);
      return values;
    });
  }

  for (const side of ['left', 'right']) for (const join of ['round', 'mitre', 'bevel']) {
    await runCase(`courtyard-line-offset-${side}-${join}`, async () => {
      await setStudy('courtyard', 'move');
      await select('curveSide', side);
      await select('join', join);
      await setSlider('offsetDistance', 4);
      const values = await waitFor('GPU result', /written \/ .* required components/);
      assert.match(values['Geometric evidence'], new RegExp(`Signed distance ${side === 'left' ? '\\+' : '-'}4 m`));
      assert.match(values['Geometric evidence'], /Raw curve \d+ rows/);
      const counts = values['GPU result'].match(/(\d+) written \/ (\d+) required components · (\d+) \/ (\d+) vertices/);
      assert.ok(counts && Number(counts[1]) === Number(counts[2]) && Number(counts[3]) === Number(counts[4]), values['GPU result']);
      return values;
    });
  }


  for (const footprint of ['courtyard', 'convex']) for (const budget of ['quick', 'resolved']) {
    await runCase(`wavefront-${footprint}-${budget}`, async () => {
      await setStudy(footprint, 'wavefront');
      await select('budget', budget);
      await setSlider('offsetDistance', 4);
      const values = await waitFor('Geometric evidence', /Required arcs \d+;/);
      if (budget === 'resolved') assert.match(values['GPU result'], /^\d+ arcs · \d+ active offset vertices$/);
      else assert.match(values['GPU result'], /^\d+ arcs · \d+ active offset vertices$|^Withheld · written\/required arcs/);
      assert.match(values['Geometric evidence'], /Requested inward offset 4 m/);
      if (footprint === 'convex' && budget === 'resolved') {
        const areaMatch = values['Geometric evidence'].match(/Convex offset contour area ([\d.]+) m²/);
        assert.ok(areaMatch, values['Geometric evidence']);
        const expectedArea = byId.convex.buffers.find(item => item.distance === -4 && item.join === 'mitre').area;
        assert.ok(Math.abs(Number(areaMatch[1]) - expectedArea) <= 0.02,
          `Convex wavefront contour area ${areaMatch[1]} versus GEOS ${expectedArea} m² (absolute tolerance 0.02 m²)`);
      }
      return values;
    });
  }

  await runCase('circle-cache-return-original', async () => {
    await setStudy('courtyard', 'circle');
    await select('budget', 'resolved');
    await setSlider('tolerance', 0.25);
    const values = await waitFor('Geometric evidence', /Radius [\d.]+ m/);
    assert.match(values['GPU result'], /Search tolerance met/);
    const match = values['Geometric evidence'].match(/Radius ([\d.]+) m/);
    assert.ok(match && Math.abs(Number(match[1]) - byId.courtyard.circleRadius) <= 0.4);
    return values;
  });
} catch (error) {
  errors.push(`harness: ${String(error)}`);
} finally {
  lastBody = page ? await page.locator('body').innerText().catch(() => '') : '';
  if (browser) await browser.close().catch(error => errors.push(`browser close: ${String(error)}`));
  await writeFile(outputFile, JSON.stringify({baseUrl, reference, reports, errors, lastBody, lastReadouts}, null, 2) + '\n');
}
assert.deepEqual(errors, [], `Footprint shape acceptance failed; report written to ${outputFile}`);
assert.ok(reports.every(item => item.passed), `Footprint shape cases failed; report written to ${outputFile}`);
console.log(`Accepted ${reports.length} footprint shape cases; report: ${outputFile}`);
