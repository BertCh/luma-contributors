// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {buildFlightMatrixWeightReference} from '../src/scenes/flows/flight-matrix-weight-reference.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/flight-matrix-weight-browser.json';
const includeEurope = process.argv.includes('--include-europe');
const reference = await buildFlightMatrixWeightReference({includeEurope});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [];
const consoleErrors = [];
let passed = false;
let failureReason = null;
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => consoleErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
  const url = new URL(baseUrl);
  url.hash = '#/story/flight-matrix/records';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {
    const showcase = globalThis.spatialAnalysisShowcase;
    return showcase?.ready || showcase?.error;
  }, null, {timeout: 90_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null, 'story must not report a load or GPU error');
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  const readoutMap = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [
    row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''
  ])));
  const optionLabels = {
    view: {matrix: 'Adjacency matrix'},
    order: {input: 'Input (A to Z)', degree: 'Busiest first', country: 'Country + hubs'},
    matrixMeasure: {pairs: 'Airport pairs', 'route-records': 'Route records'}
  };
  const setOption = async (id, value) => {
    if (id === 'view' || id === 'matrixMeasure') {
      const label = id === 'view' ? 'Show' : 'Cell value';
      await page.getByRole('radiogroup', {name: label, exact: true}).getByRole('radio', {name: optionLabels[id][value], exact: true}).click();
      return;
    }
    if (id === 'order') {
      await page.getByRole('group', {name: 'Order airports by', exact: true}).getByRole('button', {name: optionLabels[id][value], exact: true}).click();
      return;
    }
    const control = page.locator(`#all-option-${id}`);
    const tagName = await control.evaluate(element => element.tagName.toLowerCase());
    if (tagName === 'select') {
      await control.selectOption(String(value));
      return;
    }
    const role = await control.getAttribute('role');
    if (role === 'radiogroup') {
      await control.getByRole('radio', {name: optionLabels[id][value], exact: true}).click();
      return;
    }
    if (role === 'group') {
      await control.getByRole('button', {name: optionLabels[id][value], exact: true}).click();
      return;
    }
    if (role === 'spinbutton') {
      await control.focus();
      await control.press('Home');
      for (let step = 0; step < Number(value) - 7; step++) await control.press('ArrowRight');
      return;
    }
    await control.evaluate((input, optionValue) => {
      input.value = String(optionValue);
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    }, value);
  };
  const parseInteger = value => Number(value.replaceAll(',', '').match(/\d+/)?.[0] ?? NaN);

  for (const expected of reference.results) {
    await setOption('view', 'matrix');
    await setOption('order', expected.order);
    await setOption('resolutionPower', Math.log2(expected.resolution));
    if (expected.focus) await page.locator('#all-option-focus').selectOption(expected.focus);
    await setOption('zoom', expected.extent === reference.source.airports ? 1 : 12);
    await setOption('panX', 0.5);
    await setOption('panY', 0.5);
    for (const measure of ['pairs', 'route-records']) {
      await setOption('matrixMeasure', measure);
      const expectedTotal = measure === 'pairs' ? expected.pairTotal : expected.symmetricRouteRecordTotal;
      const expectedMaximum = measure === 'pairs' ? expected.maximumPairs : expected.maximumRouteRecords;
      await page.waitForFunction(({total, maximum, occupied}) => {
        const rows = [...document.querySelectorAll('.readout')];
        const value = name => rows.find(row => row.querySelector('dt')?.textContent === name)?.querySelector('dd')?.textContent ?? '';
        const integer = name => Number(value(name).replaceAll(',', '').match(/\d+/)?.[0] ?? NaN);
        return integer('Binned symmetric total') === total && integer('Maximum cell value') === maximum &&
          integer('Cells occupied') === occupied;
      }, {total: expectedTotal, maximum: expectedMaximum, occupied: expected.occupied}, {timeout: 90_000});
      const observed = await readoutMap();
      assert.equal(parseInteger(observed['Binned symmetric total']), expectedTotal, `${expected.order}/${expected.resolution}/${measure} binned sum`);
      assert.equal(parseInteger(observed['Maximum cell value']), expectedMaximum, `${expected.order}/${expected.resolution}/${measure} maximum`);
      assert.equal(parseInteger(observed['Cells occupied']), expected.occupied, `${expected.order}/${expected.resolution}/${measure} occupancy`);
      await page.getByRole('tab', {name: 'Story', exact: true}).click();
      const allReadouts = page.locator('details').filter({has: page.locator('summary .section-title', {hasText: /^All readouts$/})});
      if (!(await allReadouts.evaluate(element => element.open))) await allReadouts.locator(':scope > summary').click();
      const chart = page.locator('.readout-chart-row[data-readout-id="cellChart"]');
      const valuesToggle = chart.locator('summary', {hasText: 'Show values'});
      if (!(await valuesToggle.evaluate(element => element.parentElement?.hasAttribute('open')))) await valuesToggle.click();
      const values = await chart.locator('tbody tr').evaluateAll(rows => rows.map(row => [...row.querySelectorAll('td')].map(cell => cell.textContent)));
      const selectedBins = values.map(row => row[1] ?? '');
      const expectedBins = measure === 'pairs' ? expected.pairClassCounts : expected.routeRecordClassCounts;
      // The chart intentionally rounds counts >= 1000 to whole thousands. Totals, occupancy and
      // maxima above remain exact integer assertions; this check verifies the displayed chart.
      const expectedDisplayedBins = expectedBins.map(value => value >= 1000 ? `${Math.round(value / 1000)}k` : String(value));
      assert.deepEqual(selectedBins, expectedDisplayedBins, `${expected.order}/${expected.resolution}/${measure} displayed class bins`);
      assert.equal(parseInteger(observed['Route pairs']), reference.source.records);
      if (expected.extent === reference.source.airports && expected.order === 'input' && expected.resolution === 128) {
        assert.equal(expected.pairTotal, reference.source.records * 2, 'full symmetric pair conservation');
        assert.equal(expected.symmetricRouteRecordTotal, reference.source.routeRecordTotal * 2, 'full symmetric route-record conservation');
      }
      if (expected.focus) await page.screenshot({path: outputFile.replace(/\.json$/, '') + `-${expected.focus}-${measure}.png`, fullPage: true});
      reports.push({expected, measure, observed, selectedBins, expectedDisplayedBins});
      await page.getByRole('tab', {name: 'All controls', exact: false}).click();
    }
  }
  assert.deepEqual(consoleErrors, [], 'Browser console errors are reported separately from numerical assertions');
  passed = true;
} catch (error) {
  failureReason = error instanceof Error ? error.message : String(error);
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({
    baseUrl,
    status: passed ? 'passed' : 'failed',
    passed,
    failureReason,
    consoleErrors,
    source: reference.source,
    reports
  }, null, 2) + '\n');
}
console.log(`Flight matrix weight browser result: ${passed ? 'passed' : 'failed'} (${reports.length} checks), ${outputFile}`);
