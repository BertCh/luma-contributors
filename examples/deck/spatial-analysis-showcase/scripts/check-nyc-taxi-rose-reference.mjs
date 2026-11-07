// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/nyc-taxi-rose-reference.json';
const reference = JSON.parse(
  await readFile(
    new URL(
      '../../../../dev-docs/research/showcase-overhaul-2026-10-25/nyc-taxi-rose-reference.json',
      import.meta.url
    ),
    'utf8'
  )
);
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const reports = [];
const consoleErrors = [];

function displayHalfUnit(value) {
  const magnitude = Math.abs(value);
  if (magnitude >= 1e9) return 5e6;
  if (magnitude >= 1e6) return 5000;
  if (magnitude >= 1e4) return 50;
  if (magnitude >= 100) return 0.5;
  if (magnitude >= 10) return 0.05;
  if (magnitude >= 1) return 0.005;
  if (magnitude >= 0.001) return 0.0005;
  return 0.00005;
}

function parseDisplayed(value) {
  const match = value.replaceAll(',', '').match(/^(-?[0-9.]+)([KMB])?$/);
  assert.ok(match, `Unexpected chart number: ${value}`);
  const scale = match[2] === 'B' ? 1e9 : match[2] === 'M' ? 1e6 : match[2] === 'K' ? 1e3 : 1;
  return Number(match[1]) * scale;
}
let page;
let passed = false;
let failureReason = null;

try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => consoleErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  const url = new URL(baseUrl);
  url.hash = '#/story/nyc-taxi-trails';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {
    timeout: 90_000
  });
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  const readouts = () =>
    Object.fromEntries(
      [...document.querySelectorAll('.readout')].map(row => [
        row.querySelector('dt')?.textContent?.trim() ?? '',
        row.querySelector('dd')?.textContent?.trim() ?? ''
      ])
    );
  const setRange = async (id, value) =>
    page.locator(`#all-option-${id}`).evaluate((element, nextValue) => {
      element.value = String(nextValue);
      element.dispatchEvent(new Event('input', {bubbles: true}));
      element.dispatchEvent(new Event('change', {bubbles: true}));
    }, value);

  const playSwitch = page.locator('#all-option-play');
  if (await playSwitch.count() && await playSwitch.isChecked()) {
    await playSwitch.focus();
    await playSwitch.press('Space');
  }
  const pauseButton = page.getByRole('button', {name: 'Pause', exact: true});
  if (await pauseButton.count()) await pauseButton.click();
  await setRange('time', reference.playheadSecondsAfterDatasetOrigin);
  await page.getByRole('radiogroup', {name: 'Colour by', exact: true}).getByRole('radio', {name: 'Heading', exact: true}).click();

  for (const [caseName, expected] of Object.entries(reference.results)) {
    const [mode, weighting, grouping] = caseName.split('/');
    await page.locator('#all-option-roseMode').selectOption(mode);
    await page.locator('#all-option-roseWeighting').selectOption(weighting);
    await page.locator('#all-option-roseGrouping').selectOption(grouping);
    const expectedCharts = expected.groups.map(group => {
      const result = new Array(36).fill(0);
      for (let compassSector = 0; compassSector < 36; compassSector++) {
        const mathematicalSector = ((9 - compassSector) % group.bins.length + group.bins.length) % group.bins.length;
        result[compassSector] = mode === 'axial' ? group.bins[mathematicalSector] / 2 : group.bins[mathematicalSector];
      }
      return result;
    });
    await page.waitForFunction(({expectedValues, tolerances, activeCount, weighting}) => {
      const row = [...document.querySelectorAll('.readout')].find(
        item => item.querySelector('dt')?.textContent?.trim() === 'Active modeled route headings'
      );
      const readoutRows = [...document.querySelectorAll('.readout')];
      const support = readoutRows.find(item => item.querySelector('dt')?.textContent?.trim() === 'Rose support / weight')?.querySelector('dd')?.textContent ?? '';
      if (!support.replaceAll(',', '').includes(`${activeCount} nonzero segment headings`) || !support.includes(weighting === 'segment-length' ? 'current segments' : 'equal contribution')) return false;
      const actual = [...(row?.querySelectorAll('.chart-sectors') ?? [])].map(sectors =>
        [...sectors.querySelectorAll('title')].map(title => {
          const text = title.textContent ?? '';
          const value = text.slice(text.indexOf(':') + 1).trim().replaceAll(',', '');
          const match = value.match(/^(-?[0-9.]+)([KMB])?$/);
          if (!match) return Number.NaN;
          const scale = match[2] === 'B' ? 1e9 : match[2] === 'M' ? 1e6 : match[2] === 'K' ? 1e3 : 1;
          return Number(match[1]) * scale;
        })
      );
      if (actual.length !== expectedValues.length) return false;
      return actual.every((group, groupIndex) => group.length === expectedValues[groupIndex].length &&
        group.every((value, index) => {
          const expected = expectedValues[groupIndex][index];
          const tolerance = tolerances[groupIndex][index];
          return Number.isFinite(value) && Math.abs(value - expected) <= tolerance + 1e-5;
        }));
    }, {expectedValues: expectedCharts, tolerances: expectedCharts.map(group => group.map(value => displayHalfUnit(value) + Math.max(0.005, Math.abs(value) * 1e-6))), activeCount: expected.groups.reduce((total, group) => total + group.count, 0), weighting}, {timeout: 60_000});
    const observed = await page.evaluate(readouts);
    const support = observed['Rose support / weight'] ?? '';
    const concentration = observed['Modeled-heading concentration'] ?? '';
    const expectedCount = expected.groups.reduce((total, group) => total + group.count, 0);
    const expectedWeight = expected.groups.reduce((total, group) => total + group.totalWeight, 0);
    assert.ok(support.replaceAll(',', '').includes(`${expectedCount} nonzero segment headings`), `${caseName}: ${support}`);
    if (weighting === 'segment-length') {
      assert.ok(support.includes('current segments'), `${caseName}: ${support}`);
      assert.ok(expectedWeight > 0, `${caseName}: CPU reference has positive segment length`);
    } else {
      assert.ok(support.includes('equal contribution'), `${caseName}: ${support}`);
    }
    const expectedResultant = expected.groups.reduce((total, group) => {
      const scale = mode === 'axial' ? 2 : 1;
      const cosine = Math.cos(scale * group.meanAngle) * group.resultantLength * group.totalWeight;
      const sine = Math.sin(scale * group.meanAngle) * group.resultantLength * group.totalWeight;
      return {x: total.x + cosine, y: total.y + sine, weight: total.weight + group.totalWeight};
    }, {x: 0, y: 0, weight: 0});
    const cpuR = Math.hypot(expectedResultant.x, expectedResultant.y) / expectedResultant.weight;
    const gpuR = Number(concentration.match(/R=([0-9.]+)/)?.[1]);
    assert.ok(Number.isFinite(gpuR), `${caseName}: missing GPU resultant length: ${concentration}`);
    assert.ok(Math.abs(gpuR - cpuR) <= 0.025, `${caseName}: GPU R=${gpuR}, CPU R=${cpuR}`);

    const chart = await page.evaluate(() => {
      const row = [...document.querySelectorAll('.readout')].find(
        item => item.querySelector('dt')?.textContent?.trim() === 'Active modeled route headings'
      );
      return [...(row?.querySelectorAll('.chart-sectors') ?? [])].map(sectors =>
        [...sectors.querySelectorAll('title')].map(title => {
          const text = title.textContent ?? '';
          return text.slice(text.indexOf(':') + 1).trim();
        })
      );
    });
    assert.equal(chart.length, expectedCharts.length, `${caseName}: rose chart count`);
    for (let group = 0; group < chart.length; group++) {
      assert.equal(chart[group].length, 36, `${caseName}: chart sector count for group ${group}`);
      for (let bin = 0; bin < 36; bin++) {
        const observed = parseDisplayed(chart[group][bin]);
        assert.ok(
          Math.abs(observed - expectedCharts[group][bin]) <= displayHalfUnit(expectedCharts[group][bin]) + Math.max(0.005, Math.abs(expectedCharts[group][bin]) * 1e-6),
          `${caseName}, group ${group}, compass sector ${bin}: displayed GPU ${chart[group][bin]}, CPU ${expectedCharts[group][bin]}`
        );
      }
    }
    reports.push({caseName, expectedCount, expectedWeight, cpuR, gpuR, support, concentration});
  }
  assert.deepEqual(consoleErrors, [], 'Browser errors must not be hidden by numerical agreement');
  passed = true;
} catch (error) {
  failureReason = String(error);
  if (page) {
    await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true});
    reports.push({failure: String(error), visibleText: await page.locator('body').innerText()});
  }
  throw error;
} finally {
  await browser.close();
  await writeFile(
    outputFile,
    `${JSON.stringify({baseUrl, passed, status: passed ? 'passed' : 'failed', failureReason, reports, consoleErrors}, null, 2)}\n`
  );
}
console.log(`Verified ${reports.length} GPU rose variants: ${outputFile}`);
