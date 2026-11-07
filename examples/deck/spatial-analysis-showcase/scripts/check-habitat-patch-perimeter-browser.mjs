// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';
import {captureBuildEvidence} from './earth-build-evidence.mjs';

const oracle = JSON.parse(
  await readFile(
    new URL(
      '../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json',
      import.meta.url
    ),
    'utf8'
  )
).habitatPatchPerimeter;
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory =
  process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/habitat-patch-perimeter';
await mkdir(outputDirectory, {recursive: true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const results = [];
const errors = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => {
  if (message.type() === 'error') errors.push(message.text());
});
const select = (id, value) =>
  page.locator(`#panel-controls select[id$="option-${id}"]`).selectOption(String(value));
const readout = id => page.locator(`[data-readout="${id}"]`).first().textContent();
const waitForPerimeter = reference =>
  page.waitForFunction(
    selectedReference => {
      const text = document.querySelector('[data-readout="patchPerimeter"]')?.textContent ?? '';
      const borderText =
        selectedReference.countRasterBorder
          ? 'including outer crop faces'
          : 'excluding only outer crop faces';
      return (
        text.includes(selectedReference.label) &&
        text.includes(borderText) &&
        text.includes('WorldCover source rows')
      );
    },
    reference,
    {timeout: 180000}
  );

const setChecked = async (locator, expected) => {
  if (await locator.isChecked() !== expected) {
    await locator.focus();
    await locator.press('Space');
  }
  assert.equal(await locator.isChecked(), expected);
};

try {
  await page.goto(`${baseUrl.replace(/\/$/, '')}/#/story/habitat-connectivity/patches`, {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(
    () => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error,
    null,
    {timeout: 180000}
  );
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await installReadoutIdentifiers(page, 'habitat-connectivity');
  await page.locator('#tab-controls').click();
  await select('layer', 'patches');

  for (const countRasterBorder of [true, false]) {
    for (const group of ['tree', 'woody', 'broad']) {
      const reference = oracle.records.find(
        record => record.group === group && record.countRasterBorder === countRasterBorder
      );
      assert.ok(reference);
      await select('habitat', reference.group);
      await setChecked(page.locator('#panel-controls input[id$="option-countRasterBorder"]'), reference.countRasterBorder);
      await waitForPerimeter(reference);

      const text = await readout('patchPerimeter');
      const expectedBorderLabel = reference.countRasterBorder
        ? 'including outer crop faces'
        : 'excluding only outer crop faces';
      assert.ok(text.includes(reference.label), text);
      assert.ok(text.includes(expectedBorderLabel), text);
      assert.ok(
        text.includes(
          `WorldCover source rows ${oracle.sourceCrop.firstRow}–${oracle.sourceCrop.lastRow}, columns ${oracle.sourceCrop.firstColumn}–${oracle.sourceCrop.lastColumn}`
        ),
        text
      );
      const measurements = text.match(/([\d,]+) m total · ([\d,]+) patches/);
      assert.ok(measurements, text);
      assert.equal(Number(measurements[1].replaceAll(',', '')), reference.perimeterMeters);
      assert.equal(Number(measurements[2].replaceAll(',', '')), reference.components8);
      results.push({reference, text});
      console.log(
        `${results.length}/${oracle.records.length} ${reference.group}/${reference.countRasterBorder}`
      );
    }
  }

  for (const group of ['tree', 'woody', 'broad']) {
    const included = oracle.records.find(
      record => record.group === group && record.countRasterBorder
    );
    const excluded = oracle.records.find(
      record => record.group === group && !record.countRasterBorder
    );
    assert.equal(
      included.perimeterMeters - excluded.perimeterMeters,
      included.outerExposedFaces * oracle.sourceCrop.cellSizeMeters
    );
  }

  assert.deepEqual(errors, []);
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: true, results, errors}, null, 2));
  console.log(JSON.stringify({passed: true, checks: results.length, errors}));
} catch (error) {
  await writeFile(
    `${outputDirectory}/report.json`,
    JSON.stringify({passed: false, error: String(error), results, errors}, null, 2)
  );
  throw error;
} finally {
  await browser.close();
  await verifyBuildEvidence();
}
