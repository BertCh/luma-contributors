// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/route-editing-reference';
const reference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/cta-geometry-validation.json', import.meta.url), 'utf8'));
const translatedCoordinates = reference.inputs.source.map(([east, north]) => [Math.fround(east + 64), Math.fround(north - 32)]);
const recoveredCoordinates = translatedCoordinates.map(([east, north]) => [Math.fround(east - 64), Math.fround(north + 32)]);
const cases = [...reference.editingCases, {edit: 'translation', tolerance: 20, gridSize: 1, coordinates: translatedCoordinates}];
const lineLength = coordinates => coordinates.slice(1).reduce((total, point, index) => total + Math.hypot(point[0] - coordinates[index][0], point[1] - coordinates[index][1]), 0);

function layout(edit) {
  const inputCount = edit === 'loop-revisit' ? 8 : reference.inputs.source.length;
  const outputCapacity = edit === 'loop-revisit' ? 264 : edit === 'segment-snap' ? inputCount * (reference.inputs.stations.length + 1) : inputCount;
  const statusCount = edit === 'cleanup' ? 3 : edit === 'precision' || edit === 'segment-snap' ? 1 : edit === 'loop-revisit' ? 4 : 0;
  const wordCount = inputCount * 2 + 9 + outputCapacity * 2 + 2 + statusCount + (edit === 'segment-snap' ? 4 : 0) + (edit === 'vertex-snap' ? inputCount : 0) + (edit === 'translation' ? inputCount * 2 + 2 : 0);
  return {inputCount, outputCapacity, statusCount, wordCount};
}

function parseResult(values, edit) {
  const {inputCount, outputCapacity, statusCount, wordCount} = layout(edit);
  assert.equal(values.length, wordCount);
  const words = Uint32Array.from(values);
  const floats = new Float32Array(words.buffer);
  let cursor = 0;
  const reversed = Array.from(floats.slice(cursor, (cursor += inputCount * 2)));
  const normalized = Array.from(words.slice(cursor, (cursor += 2)));
  const equality = words[cursor++];
  const predicates = Array.from(words.slice(cursor, (cursor += 6)));
  const coordinates = Array.from(floats.slice(cursor, (cursor += outputCapacity * 2)));
  const offsets = Array.from(words.slice(cursor, (cursor += 2)));
  const status = Array.from(words.slice(cursor, (cursor += statusCount)));
  const references = edit === 'vertex-snap' ? Array.from(words.slice(cursor, (cursor += inputCount))) : [];
  const counts = edit === 'segment-snap' ? Array.from(words.slice(cursor, (cursor += 4))) : [];
  const recovered = edit === 'translation' ? Array.from(floats.slice(cursor, (cursor += inputCount * 2))) : [];
  const lengths = edit === 'translation' ? Array.from(floats.slice(cursor, (cursor += 2))) : [];
  const count = edit === 'loop-revisit' ? status[0] : edit === 'segment-snap' ? counts[0] : offsets[1];
  assert.equal(cursor, wordCount);
  return {reversed, normalized, equality, predicates, coordinates: coordinates.slice(0, count * 2), count, offsets, status, references, counts, recovered, lengths};
}

await mkdir(outputDirectory, {recursive: true});
const errors = [];
const reports = [];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
let page;
try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}, reducedMotion: 'reduce'});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  await page.addInitScript(lengths => {
    const original = GPUBuffer.prototype.getMappedRange;
    GPUBuffer.prototype.getMappedRange = function (...arguments_) {
      const bytes = original.apply(this, arguments_);
      if (lengths.includes(bytes.byteLength / 4)) {
        globalThis.__routeEditingReadback = Array.from(new Uint32Array(bytes));
        globalThis.__routeEditingSequence = (globalThis.__routeEditingSequence ?? 0) + 1;
      }
      return bytes;
    };
  }, [...new Set(cases.map(scenario => layout(scenario.edit).wordCount))]);
  const url = new URL(baseUrl);
  url.hash = '#/story/route-editing';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const scenario of cases) {
    await page.evaluate(settings => {
      for (const id of ['tolerance', 'gridSize', 'edit']) {
        const control = document.getElementById(`all-option-${id}`);
        if (control.value === String(settings[id])) continue;
        control.value = String(settings[id]);
        control.dispatchEvent(new Event(control instanceof HTMLSelectElement ? 'change' : 'input', {bubbles: true}));
      }
    }, scenario);
    const structure = layout(scenario.edit);
    await page.waitForFunction(({expected, structure, edit}) => {
      const values = globalThis.__routeEditingReadback;
      if (!values || values.length !== structure.wordCount) return false;
      const words = Uint32Array.from(values);
      const floats = new Float32Array(words.buffer);
      const start = structure.inputCount * 2 + 9;
      const statusOffset = start + structure.outputCapacity * 2 + 2;
      const count = edit === 'loop-revisit' ? words[statusOffset] : edit === 'segment-snap' ? words[statusOffset + 1] : words[statusOffset - 1];
      return count === expected.length && expected.flat().every((value, index) => Math.abs(value - floats[start + index]) < 0.005);
    }, {expected: scenario.coordinates, structure, edit: scenario.edit}, {timeout: 60_000});
    const observed = parseResult(await page.evaluate(() => globalThis.__routeEditingReadback), scenario.edit);
    const source = scenario.edit === 'loop-revisit' ? [0, 1, 2, 3, 2, 3, 4, 5].map(index => reference.inputs.source[index]) : reference.inputs.source;
    assert.deepEqual(observed.reversed, source.toReversed().flat(), 'Orientation reverses exact stored rows');
    assert.deepEqual(observed.normalized, [0, 0]);
    assert.equal(observed.equality, 1);
    assert.deepEqual(observed.predicates.slice(0, 4), [0, 0, scenario.edit === 'loop-revisit' ? 0 : 1, 0]);
    assert.equal(observed.predicates[4], 0, 'Predicate completeness must hold');
    if (scenario.edit === 'cleanup') assert.deepEqual(observed.status, [observed.count, 0, observed.count]);
    if (scenario.edit === 'precision') assert.deepEqual(observed.status, [0]);
    if (scenario.edit === 'vertex-snap') assert.deepEqual(observed.references, scenario.referenceRows);
    if (scenario.edit === 'segment-snap') {
      assert.deepEqual(observed.status, [0]);
      assert.deepEqual(observed.counts, [observed.count, observed.count, 0, 0]);
    }
    if (scenario.edit === 'loop-revisit') assert.deepEqual(observed.status, [observed.count, observed.count, 0, 0]);
    if (scenario.edit === 'translation') {
      assert.deepEqual(observed.coordinates, translatedCoordinates.flat(), 'Every declared translated coordinate matches float32 arithmetic');
      assert.deepEqual(observed.recovered, recoveredCoordinates.flat(), 'Every inverse-transformed row retains original order');
      assert.deepEqual(observed.offsets, [0, source.length]);
      assert.ok(Math.abs(observed.lengths[0] - lineLength(source)) < 0.05, 'GPU source length agrees with independent local-coordinate length within 5 cm');
      assert.ok(Math.abs(observed.lengths[1] - lineLength(translatedCoordinates)) < 0.05, 'GPU translated length agrees with independent local-coordinate length within 5 cm');
      assert.ok(Math.abs(observed.lengths[1] - observed.lengths[0]) < 0.05, 'Translation preserves measured length within float32 reduction tolerance');
      const maximumRecoveryError = Math.max(...recoveredCoordinates.map((point, index) => Math.hypot(point[0] - source[index][0], point[1] - source[index][1])));
      assert.ok(maximumRecoveryError < 0.0001);
      await page.locator('#all-option-focus').selectOption('downtown');
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      await page.screenshot({path: `${outputDirectory}/translation.png`});
    }
    reports.push({edit: scenario.edit, tolerance: scenario.tolerance, gridSize: scenario.gridSize, observed});
  }
  await page.locator('#all-option-edit').selectOption('loop-revisit');
  await page.waitForFunction(wordCount => globalThis.__routeEditingReadback?.length === wordCount, layout('loop-revisit').wordCount, {timeout: 60_000});
  await page.locator('#all-option-focus').selectOption('revisit');
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await page.screenshot({path: `${outputDirectory}/revisit.png`});
  assert.deepEqual(errors, []);
} catch (error) {
  await writeFile(`${outputDirectory}/failure.txt`, String(error.stack ?? error));
  await page?.screenshot({path: `${outputDirectory}/failure.png`}).catch(() => {});
  throw error;
} finally {
  const lastReadback = await page?.evaluate(() => globalThis.__routeEditingReadback ?? null).catch(() => null);
  await writeFile(`${outputDirectory}/last-readback.json`, JSON.stringify(lastReadback ?? null) + '\n');
  await browser.close();
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({reports, errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} route-editing cases against independent GEOS/direct references: ${outputDirectory}`);
