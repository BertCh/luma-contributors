// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';

const oracle = JSON.parse(
  await readFile(
    new URL(
      '../../../../dev-docs/research/showcase-overhaul-2026-10-25/fire-policy-reference.json',
      import.meta.url
    ),
    'utf8'
  )
);
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory =
  process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/fire-policy-reference';
await mkdir(outputDirectory, {recursive: true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await page.addInitScript(() => {
  globalThis.firePolicyGPUErrors = [];
  if (typeof GPUAdapter === 'undefined') return;
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (...arguments_) {
    const device = await requestDevice.apply(this, arguments_);
    device.addEventListener('uncapturederror', event => {
      globalThis.firePolicyGPUErrors.push(String(event.error));
    });
    return device;
  };
});
const errors = [];
const results = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => {
  if (message.type() === 'error') errors.push(message.text());
});

const selectControl = (identifier, value) =>
  page.locator(`#panel-controls select[id$="option-${identifier}"]`).selectOption(String(value));
const setToggle = async (identifier, value) => {
  const toggle = page.locator(`#panel-controls input[id$="option-${identifier}"]`);
  if ((await toggle.isChecked()) !== value) {
    await toggle.focus();
    await toggle.press('Space');
  }
};
const setSlider = (identifier, value) =>
  page.locator(`#panel-controls input[id$="option-${identifier}"]`).evaluate((input, nextValue) => {
    input.value = String(nextValue);
    input.dispatchEvent(new Event('input', {bubbles: true}));
  }, value);
const readout = identifier => page.locator(`[data-readout="${identifier}"]`).last();
const readoutText = async identifier => (await readout(identifier).textContent()) ?? '';
const waitForReadouts = expected =>
  page.waitForFunction(
    values =>
      Object.entries(values).every(([identifier, expectedValue]) =>
        (document.querySelectorAll(`[data-readout="${identifier}"]`).item(
          document.querySelectorAll(`[data-readout="${identifier}"]`).length - 1
        )?.textContent ?? '').includes(expectedValue)
      ),
    expected,
    {timeout: 180000}
  );
const parseCount = text => Number(text.replaceAll(',', ''));

function verifyFocusMapping(text, connectivity, mergeReference) {
  const match = text.match(
    /^([\d,]+) labeled cells · checksum ([\d,]+) · cells by marker \[([\d, ]+)\] · mapping (.+) · (fixed|adaptive), Δmean ≤ ([\d.]+)$/
  );
  assert.ok(match, `Unexpected detailed basin readout: ${text}`);
  const reference = oracle.focus[connectivity];
  const expectedMapping = mergeReference.mapping.map(root => root + 1);
  const actualMapping = Array.from(match[4].matchAll(/(\d+)→(\d+)/g), mapping => [
    Number(mapping[1]),
    Number(mapping[2])
  ]);
  assert.equal(parseCount(match[1]), reference.labeledCells, 'GPU watershed label count');
  assert.equal(parseCount(match[2]), reference.weightedLabelSum, 'GPU watershed label checksum');
  assert.deepEqual(
    match[3].split(', ').map(Number),
    reference.labelCounts,
    'GPU cells by watershed marker'
  );
  assert.deepEqual(
    actualMapping,
    expectedMapping.map((root, index) => [index + 1, root]),
    'GPU region mapping (displayed one-based; oracle is zero-based)'
  );
  assert.equal(match[5], mergeReference.mode, 'selected GPU merge mode');
  assert.equal(Number(match[6]), mergeReference.threshold, 'selected GPU merge threshold');
}

try {
  await page.goto(`${baseUrl.replace(/\/$/, '')}/#/story/fire-scar-topology/threshold`, {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(
    () => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error,
    null,
    {timeout: 180000}
  );
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await installReadoutIdentifiers(page, 'fire-scar-topology', 'raster');
  await page.locator('#tab-controls').click();
  await selectControl('mapLayer', 'threshold');
  await selectControl('cleanup', 'none');
  await selectControl('thresholdMethod', 'fixed');

  const manualCases = [
    ...[true, false].flatMap(inclusive =>
      ['above', 'below', 'range'].map(operation => ({operation, inclusive}))
    )
  ];
  for (const reference of manualCases) {
    const lower = 0.27;
    const upper = 0.66;
    const sliderCutoff = lower;
    await selectControl('thresholdOperation', reference.operation);
    if (reference.operation !== 'range') await selectControl('thresholdMethod', 'fixed');
    await setToggle('thresholdInclusive', reference.inclusive);
    await setSlider('thresholdValue', sliderCutoff);
    if (reference.operation === 'range') await setSlider('rangeEndpoint', upper);
    const selectionReference = oracle.selection.find(
      selection =>
        selection.operation === reference.operation &&
        selection.inclusive === reference.inclusive &&
        selection.lower === lower &&
        selection.upper === upper
    );
    assert.ok(selectionReference, `Missing source oracle for ${JSON.stringify(reference)}`);
    const expectedActive =
      reference.operation === 'range'
        ? `manual ${reference.inclusive ? 'inclusive' : 'exclusive'} [0.27, 0.66]`
        : `${reference.operation} ${reference.inclusive ? 'inclusive' : 'exclusive'} · ${sliderCutoff.toFixed(2)}`;
    await waitForReadouts({
      activeThreshold: expectedActive,
      thresholdCells: `${selectionReference.cells.toLocaleString('en-US')} valid cells selected`,
      validPixels: `${oracle.supportCells.toLocaleString('en-US')} /`
    });
    const text = await readoutText('thresholdCells');
    assert.equal(parseCount(text.match(/[\d,]+/)?.[0] ?? ''), selectionReference.cells, text);
    assert.equal(await page.locator('#panel-controls select[id$="option-thresholdOperation"]').inputValue(), reference.operation);
    assert.equal(await page.locator('#panel-controls input[id$="option-thresholdInclusive"]').isChecked(), reference.inclusive);
    results.push({kind: 'manual-selection', ...reference, cells: selectionReference.cells, active: expectedActive});
    console.log(`PASS fire selection ${results.length}/22: ${expectedActive} → ${selectionReference.cells}`);
  }

  for (const operation of ['above', 'below']) {
    for (const inclusive of [true, false]) {
      const reference = oracle.sourceAnchor;
      await selectControl('thresholdOperation', operation);
      await selectControl('thresholdMethod', 'source-probe');
      await setToggle('thresholdInclusive', inclusive);
      const cells =
        operation === 'above'
          ? inclusive
            ? reference.aboveInclusive
            : reference.aboveExclusive
          : inclusive
            ? reference.belowInclusive
            : reference.belowExclusive;
      const expectedActive = `GPU source-cell threshold · row ${reference.index.toLocaleString('en-US')}`;
      await waitForReadouts({
        activeThreshold: `${operation} ${inclusive ? 'inclusive' : 'exclusive'} · ${expectedActive}`,
        thresholdCells: `${cells.toLocaleString('en-US')} valid cells selected`,
        validPixels: `${oracle.supportCells.toLocaleString('en-US')} /`
      });
      const text = await readoutText('thresholdCells');
      assert.equal(parseCount(text.match(/[\d,]+/)?.[0] ?? ''), cells, text);
      results.push({kind: 'source-equality', operation, inclusive, cells, sourceRow: reference.index});
      console.log(`PASS source-cell equality ${results.length}/22: ${operation}/${inclusive} → ${cells}`);
    }
  }

  await page.evaluate(() => {
    window.location.hash = '#/story/fire-scar-topology/basin-merging';
  });
  await page.waitForFunction(
    () => document.querySelector('#panel-story .step-row[aria-expanded="true"] .step-row-title')?.textContent === 'Compare basin merge rules',
    null,
    {timeout: 180000}
  );
  await page.locator('#tab-controls').click();
  await selectControl('mapLayer', 'watershedMerged');
  await selectControl('cleanup', 'none');
  await setSlider('regionMergeThreshold', 0.2);

  for (const connectivity of ['4', '8']) {
    for (const mode of ['fixed', 'adaptive']) {
      for (const rounds of [1, 4, 16]) {
        await selectControl('connectivity', connectivity);
        await selectControl('regionMergeMode', mode);
        await setSlider('regionMergeRounds', rounds);
        const connectivityReference = oracle.focus[connectivity];
        const mergeReference = connectivityReference.merges.find(
          merge => merge.mode === mode && merge.threshold === 0.2 && merge.rounds === rounds
        );
        assert.ok(mergeReference, `Missing focus oracle for ${connectivity}/${mode}/${rounds}`);
        const pendingStatus = mergeReference.pending
          ? `complete queue · merge incomplete at ${rounds} rounds`
          : 'complete queue · no eligible merge remains';
        await waitForReadouts({
          focusSupport: '3,647 / 4096 valid cells',
          focusMarkers: `${Object.keys(connectivityReference.markers).length}`,
          focusRegionMapping: `mapping ${mergeReference.mapping.map((root, index) => `${index + 1}→${root + 1}`).join(' ')} · ${mode}, Δmean ≤ 0.200`,
          mergedBasinCount: `${mergeReference.components} / ${Object.keys(connectivityReference.markers).length} roots`,
          watershedStatus: pendingStatus
        });
        const mappingText = await readoutText('focusRegionMapping');
        verifyFocusMapping(mappingText, connectivity, mergeReference);
        assert.equal(await page.locator('#panel-controls select[id$="option-connectivity"]').inputValue(), connectivity);
        assert.equal(await page.locator('#panel-controls select[id$="option-regionMergeMode"]').inputValue(), mode);
        assert.equal(await page.locator('#panel-controls input[id$="option-regionMergeRounds"]').inputValue(), String(rounds));
        const result = {
          kind: 'basin-merge',
          connectivity,
          mode,
          threshold: mergeReference.threshold,
          rounds,
          roots: mergeReference.components,
          pending: mergeReference.pending,
          labeledCells: connectivityReference.labeledCells,
          weightedLabelSum: connectivityReference.weightedLabelSum,
          mapping: mergeReference.mapping
        };
        results.push(result);
        if (connectivity === '4' && rounds === 16) {
          await page.screenshot({path: `${outputDirectory}/merge-${mode}.png`});
        }
        console.log(
          `PASS basin merge ${results.length}/22: ${connectivity}-neighbor/${mode}/${rounds} rounds → ${mergeReference.components} roots`
        );
      }
    }
  }

  assert.equal(oracle.supportCells, 35583, 'source common-support oracle');
  assert.equal(results.length, 22, 'bounded verification case count');
  assert.deepEqual(errors, [], 'Browser or GPU console errors');
  const gpuErrors = await page.evaluate(() => globalThis.firePolicyGPUErrors);
  assert.deepEqual(gpuErrors, [], 'Uncaptured WebGPU validation errors');
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: true, results, errors, gpuErrors}, null, 2));
  console.log(JSON.stringify({passed: true, checks: results.length, errors}, null, 2));
} catch (error) {
  const diagnostics = await page.evaluate(() => ({
    url: window.location.href,
    ready: globalThis.spatialAnalysisShowcase?.ready,
    appError: globalThis.spatialAnalysisShowcase?.error,
    activeStep: document.querySelector('#panel-story .step-row[aria-expanded="true"] .step-row-title')?.textContent,
    visibleReadouts: Array.from(
      document.querySelectorAll('#panel-story .step-detail:not([hidden]) [data-readout]'),
      element => [element.getAttribute('data-readout'), element.textContent]
    ),
    controls: Array.from(document.querySelectorAll('#panel-controls select, #panel-controls input'), element => [element.id, element.value])
  }));
  diagnostics.gpuErrors = await page.evaluate(() => globalThis.firePolicyGPUErrors);
  await page.screenshot({path: `${outputDirectory}/failure.png`, fullPage: true});
  await writeFile(
    `${outputDirectory}/report.json`,
    JSON.stringify({passed: false, error: String(error), results, errors, diagnostics}, null, 2)
  );
  throw error;
} finally {
  await browser.close();
  await verifyBuildEvidence();
}
