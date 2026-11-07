// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import * as h3 from 'h3-js';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {loadBixiCellFlowReference} from './bixi-cell-flow-reference.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/bixi-cell-flows-reference';
const references = await Promise.all([6, 7, 8].map(loadBixiCellFlowReference));
const wordCount = reference => reference.summary.cellCount * 76 + reference.summary.pairCount + 1;
const keyAt = (words, index) => ((BigInt(words[index + 1]) << 32n) | BigInt(words[index])).toString(16);
const sum = values => values.reduce((total, value) => total + value, 0);

function verifyOutput(values, reference) {
  assert.equal(values.length, wordCount(reference));
  const words = Uint32Array.from(values);
  const floats = new Float32Array(words.buffer);
  const cellCount = reference.summary.cellCount;
  const pairCount = reference.summary.pairCount;
  let cursor = cellCount * 2;
  const readFloat = length => floats.slice(cursor, (cursor += length));
  const readWords = length => words.slice(cursor, (cursor += length));
  const cellKeys = Array.from({length: cellCount}, (_, row) => keyAt(words, row * 2));
  assert.deepEqual(cellKeys, [...reference.cells.keys()].sort());
  const outgoing = readFloat(cellCount);
  const incoming = readFloat(cellCount);
  const directedEdges = readWords(cellCount * 12);
  const neighbors = readWords(cellCount * 12);
  const edgeFlows = readFloat(cellCount * 6);
  const boundaries = readFloat(cellCount * 36);
  const boundaryCounts = readWords(cellCount * 6);
  const validity = readWords(pairCount);
  const [invalidCount] = readWords(1);
  assert.equal(cursor, values.length);
  assert.equal(invalidCount, 0);
  assert.ok(validity.every(value => value === 1));
  assert.equal(sum(outgoing), reference.summary.retainedTotal);
  assert.equal(sum(incoming), reference.summary.retainedTotal);
  assert.equal(sum(edgeFlows), reference.summary.neighboring);
  let maximumBoundaryErrorMeters = 0;
  for (let row = 0; row < cellCount; row++) {
    const cell = cellKeys[row];
    assert.equal(outgoing[row], reference.cells.get(cell).outgoing, `Outgoing marginal for ${cell}`);
    assert.equal(incoming[row], reference.cells.get(cell).incoming, `Incoming marginal for ${cell}`);
    const observedEdges = [];
    for (let slot = 0; slot < 6; slot++) {
      const edgeRow = row * 6 + slot;
      const edge = keyAt(directedEdges, edgeRow * 2);
      const neighbor = keyAt(neighbors, edgeRow * 2);
      if (edge === '0') {
        assert.equal(neighbor, '0');
        assert.equal(boundaryCounts[edgeRow], 0);
        assert.equal(edgeFlows[edgeRow], 0);
        continue;
      }
      observedEdges.push(edge);
      assert.equal(h3.getDirectedEdgeOrigin(edge), cell);
      assert.equal(h3.getDirectedEdgeDestination(edge), neighbor);
      assert.equal(edgeFlows[edgeRow], reference.pairTotals.get(`${cell}/${neighbor}`) ?? 0);
      const expectedBoundary = h3.directedEdgeToBoundary(edge);
      const count = boundaryCounts[edgeRow];
      assert.equal(count, expectedBoundary.length);
      const observedBoundary = Array.from({length: count}, (_, point) => {
        const offset = edgeRow * 6 + point * 2;
        return [boundaries[offset + 1], boundaries[offset]];
      });
      const forwardError = Math.max(...observedBoundary.map((point, index) => h3.greatCircleDistance(point, expectedBoundary[index], 'km') * 1000));
      const reverseError = Math.max(...observedBoundary.map((point, index) => h3.greatCircleDistance(point, expectedBoundary[count - 1 - index], 'km') * 1000));
      const error = Math.min(forwardError, reverseError);
      maximumBoundaryErrorMeters = Math.max(maximumBoundaryErrorMeters, error);
      assert.ok(error < 2, `Actual GPU edge boundary ${edge} must be within 2 m of independent H3 geometry; got ${error} m`);
    }
    assert.deepEqual(observedEdges.sort(), h3.originToDirectedEdges(cell).sort());
  }
  return {...reference.summary, gpuOutgoing: sum(outgoing), gpuIncoming: sum(incoming), gpuEdgeFlows: sum(edgeFlows), invalidCount, maximumBoundaryErrorMeters};
}

await mkdir(outputDirectory, {recursive: true});
const reports = [];
const errors = [];
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
      if (lengths.includes(bytes.byteLength / 4)) globalThis.__bixiCellFlowReadback = Array.from(new Uint32Array(bytes));
      return bytes;
    };
  }, references.map(wordCount));
  const url = new URL(baseUrl);
  url.hash = '#/story/bixi-cell-flows';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const reference of references) {
    await page.locator('#all-option-resolution').selectOption(String(reference.summary.resolution));
    await page.waitForFunction(length => globalThis.__bixiCellFlowReadback?.length === length, wordCount(reference), {timeout: 90_000});
    const values = await page.evaluate(() => globalThis.__bixiCellFlowReadback);
    await writeFile(`${outputDirectory}/resolution-${reference.summary.resolution}-readback.json`, JSON.stringify(values) + '\n');
    reports.push(verifyOutput(values, reference));
    await page.locator('#all-option-display').selectOption('neighbor-boundaries');
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await page.screenshot({path: `${outputDirectory}/resolution-${reference.summary.resolution}.png`});
  }
  assert.deepEqual(errors, []);
} catch (error) {
  await writeFile(`${outputDirectory}/failure.txt`, String(error.stack ?? error));
  await page?.screenshot({path: `${outputDirectory}/failure.png`}).catch(() => {});
  throw error;
} finally {
  await browser.close();
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({reports, errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} BIXI resolutions: every marginal, edge identity/flow and GPU boundary, with retained-OD conservation: ${outputDirectory}`);
