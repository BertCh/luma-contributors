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
const outputFile = process.argv[3] ?? '/tmp/earthquake-inhomogeneous-browser-reference.json';
const referenceFile = outputFile.replace(/\.json$/, '-cpu.json');
const referenceScript = fileURLToPath(new URL('./earthquake-inhomogeneous-reference.mjs', import.meta.url));
execFileSync(process.execPath, [referenceScript, referenceFile], {stdio: 'inherit'});
const reference = JSON.parse(await readFile(referenceFile, 'utf8'));
const reports = [];
const errors = [];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
let page;
let failure;

function failWithPageContext(message, readouts, bodyText) {
  return new Error(
    `${message}\nAll readouts (including hidden): ${JSON.stringify(readouts, null, 2)}\nBody:\n${bodyText}`
  );
}

async function getReadouts() {
  return page.evaluate(() =>
    Object.fromEntries(
      [...document.querySelectorAll('.readout')].map(row => [
        row.querySelector('dt')?.textContent?.trim() ?? '',
        row.querySelector('dd')?.textContent?.trim() ?? ''
      ])
    )
  );
}

async function assertControlApplicability(expectation) {
  const controls = {
    intensityBandwidth: '#all-option-intensityBandwidth',
    selfContribution: '#all-option-selfContribution',
    correction: '#all-option-inhomogeneousCorrection',
    pairEstimator: '#all-option-pairCorrelationEstimator',
    pairBandwidth: '#all-option-pairCorrelationBandwidth',
    pairDivisor: '#all-option-pairCorrelationDivisor',
    zeroCorrection: '#all-option-zeroCorrection'
  };
  for (const [name, selector] of Object.entries(controls)) {
    const disabled = await page.locator(selector).isDisabled();
    assert.equal(disabled, expectation[name], `${name} disabled applicability`);
  }
}

async function assertScenarioMetadata(scenario, status) {
  const actual = await page.evaluate(() => {
    const readValue = name => document.querySelector(`#all-option-${name}`)?.value ?? null;
    return {
      mode: readValue('mode'),
      intensityModel: readValue('intensityModel'),
      intensityBandwidth: readValue('intensityBandwidth'),
      selfContribution: readValue('selfContribution'),
      inhomogeneousCorrection: readValue('inhomogeneousCorrection'),
      pairCorrelationEstimator: readValue('pairCorrelationEstimator'),
      pairCorrelationBandwidth: readValue('pairCorrelationBandwidth'),
      pairCorrelationDivisor: readValue('pairCorrelationDivisor'),
      zeroCorrection: readValue('zeroCorrection')
    };
  });
  assert.equal(actual.mode, 'inhomogeneous');
  assert.equal(actual.intensityModel, scenario.options.model);
  assert.equal(actual.inhomogeneousCorrection, scenario.options.correction);
  if (scenario.options.model === 'epanechnikov') {
    assert.equal(Number(actual.intensityBandwidth), scenario.options.intensityBandwidth);
    assert.equal(actual.selfContribution, scenario.options.self);
    assert.ok(status.includes(`Epanechnikov intensity bandwidth ${scenario.options.intensityBandwidth / 1000} km`));
  } else {
    assert.match(status, /constant λ = n \/ area/);
  }
  if (scenario.options.correction === 'translation') {
    assert.equal(actual.pairCorrelationEstimator, scenario.options.estimator);
    if (scenario.options.estimator === 'epanechnikov') {
      assert.equal(Number(actual.pairCorrelationBandwidth), scenario.options.pairBandwidth);
      assert.equal(actual.pairCorrelationDivisor, scenario.options.divisor);
      assert.equal(actual.zeroCorrection, scenario.options.zeroCorrection);
    }
  }
  assert.ok(status.includes(`${scenario.options.correction} correction`));
}

async function chooseScenario(scenario) {
  const previousRebuild = await page.evaluate(() => globalThis.spatialAnalysisShowcase?.rebuildCount ?? 0);
  let changed = false;
  const select = async (name, value) => {
    if (value !== undefined && await page.locator(`#all-option-${name}`).inputValue() !== String(value)) {
      changed = true;
      await page.locator(`#all-option-${name}`).selectOption(String(value));
    }
  };
  const setSlider = async (name, value) => {
    if (value === undefined || await page.locator(`#all-option-${name}`).inputValue() === String(value)) return;
    changed = true;
    await page.locator(`#all-option-${name}`).evaluate((input, nextValue) => {
      input.value = String(nextValue);
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    }, value);
  };
  await select('mode', 'inhomogeneous');
  await select('intensityModel', scenario.model);
  if (scenario.model === 'epanechnikov') {
    await setSlider('intensityBandwidth', scenario.intensityBandwidth);
    await select('selfContribution', scenario.self);
  }
  await select('inhomogeneousCorrection', scenario.correction);
  if (scenario.correction === 'translation') {
    await select('pairCorrelationEstimator', scenario.estimator);
    if (scenario.estimator === 'epanechnikov') {
      await setSlider('pairCorrelationBandwidth', scenario.pairBandwidth);
      await select('pairCorrelationDivisor', scenario.divisor);
      await select('zeroCorrection', scenario.zeroCorrection);
    }
  }
  if (changed) await page.waitForFunction(previous => (globalThis.spatialAnalysisShowcase?.ready && globalThis.spatialAnalysisShowcase.rebuildCount > previous) || document.body.innerText.includes('This analysis could not start'), previousRebuild, {timeout: 120000});
  assert.ok(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), await page.locator('body').innerText());
}

function compareClose(actual, expected, label) {
  const tolerance = Math.max(0.0001, Math.abs(expected) * 0.0015);
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} vs ${expected} (±${tolerance})`);
}

try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  const url = new URL(baseUrl);
  url.hash = '#/story/earthquake-pair-structure';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () =>
      globalThis.spatialAnalysisShowcase?.ready ||
      document.body.innerText.includes('This analysis could not start'),
    null,
    {timeout: 120000}
  );
  assert.ok(
    await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready),
    await page.locator('body').innerText()
  );
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();

  const firstScenario = reference.scenarios[0];
  await chooseScenario(firstScenario.options);
  await assertControlApplicability({
    intensityBandwidth: true,
    selfContribution: true,
    correction: false,
    pairEstimator: false,
    pairBandwidth: true,
    pairDivisor: true,
    zeroCorrection: true
  });

  for (const scenario of reference.scenarios) {
    await chooseScenario(scenario.options);
    const estimatedIntensity = scenario.options.model === 'epanechnikov';
    const smoothedPairs = scenario.options.estimator === 'epanechnikov';
    await assertControlApplicability({
      intensityBandwidth: !estimatedIntensity,
      selfContribution: !estimatedIntensity,
      correction: false,
      pairEstimator: scenario.options.correction !== 'translation',
      pairBandwidth: scenario.options.correction !== 'translation' || !smoothedPairs,
      pairDivisor: scenario.options.correction !== 'translation' || !smoothedPairs,
      zeroCorrection: scenario.options.correction !== 'translation' || !smoothedPairs
    });
    await page.waitForFunction(
      () => {
        const rows = [...document.querySelectorAll('.readout')];
        const status = rows.find(row => row.querySelector('dt')?.textContent?.trim() === 'Inhomogeneous GPU status')
          ?.querySelector('dd')?.textContent?.trim() ?? '';
        return status.startsWith('GPU statuses valid at all 16 radii;') ||
          status.startsWith('Inference withheld: r1: invalid input or nonfinite accumulation');
      },
      null,
      {timeout: 120000}
    );
    const readouts = await getReadouts();
    await assertScenarioMetadata(scenario, readouts['Inhomogeneous GPU status'] ?? '');
    const terminalText = readouts['GPU inhomogeneous values at 80 km'] ?? '';
    if (scenario.terminal.valid === false) {
      assert.match(readouts['Inhomogeneous GPU status'], /Inference withheld: r1: invalid input or nonfinite accumulation/);
      assert.match(terminalText, /Terminal-radius values withheld because GPU output is invalid/);
      reports.push({
        name: scenario.name,
        options: scenario.options,
        expectedInvalidReason: 'Coincident distinct events are invalid with the distance divisor.',
        status: readouts['Inhomogeneous GPU status'],
        terminal: terminalText
      });
      console.log(`Accepted invalid-distance status ${scenario.name}`);
      continue;
    }
    const match = terminalText.match(/At 80 km: K=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) m²; L−r=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) m; g=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/);
    assert.ok(match, failWithPageContext(`Terminal output is pending or invalid for ${scenario.name}`, readouts, await page.locator('body').innerText()));
    const actual = {k: Number(match[1]), lMinusR: Number(match[2]), pairCorrelation: Number(match[3])};
    compareClose(actual.k, scenario.terminal.k, `${scenario.name} K`);
    compareClose(actual.lMinusR, scenario.terminal.lMinusR, `${scenario.name} L-r`);
    compareClose(actual.pairCorrelation, scenario.terminal.pairCorrelation, `${scenario.name} g`);
    const firstRadiusMatch = terminalText.match(/At 5 km: K=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) m²; L−r=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) m; g=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/);
    assert.ok(firstRadiusMatch, 'Actual first-radius values are required to verify reflection at zero');
    const firstRadiusActual = {k: Number(firstRadiusMatch[1]), lMinusR: Number(firstRadiusMatch[2]), pairCorrelation: Number(firstRadiusMatch[3])};
    compareClose(firstRadiusActual.k, scenario.bins[0].k, `${scenario.name} 5km K`);
    compareClose(firstRadiusActual.lMinusR, scenario.bins[0].lMinusR, `${scenario.name} 5km L-r`);
    compareClose(firstRadiusActual.pairCorrelation, scenario.bins[0].pairCorrelation, `${scenario.name} 5km g`);
    assert.match(readouts['Inhomogeneous GPU status'], /GPU statuses valid at all 16 radii/);
    assert.match(
      (readouts['Inhomogeneous estimator'] ?? '').toLowerCase(),
      new RegExp(`${scenario.options.correction} correction`)
    );
    reports.push({
      name: scenario.name,
      options: scenario.options,
      expected: scenario.terminal,
      actual,
      firstRadiusExpected: scenario.bins[0],
      firstRadiusActual,
      status: readouts['Inhomogeneous GPU status']
    });
    console.log(`Accepted ${scenario.name}`);
  }

  // A second request for an earlier combination must restore the correct cached graph result.
  await chooseScenario(firstScenario.options);
  await page.waitForFunction(
    () => {
      const rows = [...document.querySelectorAll('.readout')];
      const status = rows.find(row => row.querySelector('dt')?.textContent?.trim() === 'Inhomogeneous GPU status')
        ?.querySelector('dd')?.textContent?.trim() ?? '';
      const terminal = rows.find(row => row.querySelector('dt')?.textContent?.trim() === 'GPU inhomogeneous values at 80 km')
        ?.querySelector('dd')?.textContent?.trim() ?? '';
      return status.startsWith('GPU statuses valid at all 16 radii;') && terminal.startsWith('At 80 km:');
    },
    null,
    {timeout: 120000}
  );
  const cacheReadouts = await getReadouts();
  const cacheMatch = cacheReadouts['GPU inhomogeneous values at 80 km'].match(
    /At 80 km: K=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) m²; L−r=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) m; g=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/
  );
  assert.ok(cacheMatch, failWithPageContext('Cached terminal output was invalid', cacheReadouts, await page.locator('body').innerText()));
  compareClose(Number(cacheMatch[1]), firstScenario.terminal.k, 'cached K');
  compareClose(Number(cacheMatch[2]), firstScenario.terminal.lMinusR, 'cached L-r');
  compareClose(Number(cacheMatch[3]), firstScenario.terminal.pairCorrelation, 'cached g');
  reports.push({name: 'cache-return-constant-translation', passed: true});
  assert.deepEqual(errors, []);
} catch (error) {
  const details = page
    ? `\nAll readouts (including hidden): ${JSON.stringify(await getReadouts().catch(() => ({})), null, 2)}\nBody:\n${await page.locator('body').innerText().catch(() => '')}`
    : '';
  failure = new Error(`${String(error.stack ?? error)}${details}`);
} finally {
  const lastReadouts = page ? await getReadouts().catch(() => ({})) : {};
  const lastBody = page ? await page.locator('body').innerText().catch(() => '') : '';
  await browser.close();
  const report = {
    baseUrl,
    referenceFile,
    plannedCaseCount: reference.plannedCaseCount + 1,
    completedCaseCount: reports.length,
    reports,
    errors,
    lastReadouts,
    lastBody,
    failure: failure ? String(failure.stack ?? failure) : null
  };
  await writeFile(outputFile, JSON.stringify(report, null, 2) + '\n');
}
if (failure) throw failure;
console.log(`Wrote ${outputFile}`);

