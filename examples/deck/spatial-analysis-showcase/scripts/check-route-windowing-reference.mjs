// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/route-windowing-reference';
const python = process.env.CTA_REFERENCE_PYTHON ?? 'python3';
const referenceScript = fileURLToPath(new URL('./validate-cta-geometry.py', import.meta.url));
const summaryWordCount = 13_925;
const subdivisionCapacity = 256;
const subdivisionVertices = 16;
const clipVertexCapacity = 608;
const clipPieceCapacity = 1120;

function parseResult(values, method) {
  const words = Uint32Array.from(values);
  const floats = new Float32Array(words.buffer);
  let cursor = 0;
  const readFloat = count => Array.from(floats.slice(cursor, (cursor += count)));
  const readUint = count => Array.from(words.slice(cursor, (cursor += count)));
  const pairs = coordinates => Array.from({length: coordinates.length / 2}, (_, index) => coordinates.slice(index * 2, index * 2 + 2));
  const corners = pairs(readFloat(8));
  const sizes = readFloat(4);
  const longestLine = pairs(readFloat(4));
  const [diameter] = readFloat(1);
  const [hullSize, minimumOverflow] = readUint(2);
  const subdivisionCoordinates = readFloat(subdivisionCapacity * subdivisionVertices * 2);
  const subdivisionMetadata = readUint(subdivisionCapacity * 4);
  const subdivisionStatus = readUint(4);
  const subdivisionPolygons = Array.from({length: subdivisionStatus[0]}, (_, piece) => {
    const count = subdivisionMetadata[piece * 4];
    assert.ok(count >= 3 && count <= subdivisionVertices);
    assert.equal(subdivisionMetadata[piece * 4 + 1], 0, 'Every piece retains the source feature');
    assert.equal(subdivisionMetadata[piece * 4 + 2], 0, 'Every piece retains the source polygon');
    const start = piece * subdivisionVertices * 2;
    return pairs(subdivisionCoordinates.slice(start, start + count * 2));
  });
  function readClip() {
    const coordinates = readFloat(clipVertexCapacity * 2);
    const offsets = readUint(clipPieceCapacity + 1);
    const [count, vertexCount, overflow, required, uncertain, candidateOverflow] = readUint(6);
    assert.ok(count <= clipPieceCapacity && vertexCount <= clipVertexCapacity);
    const paths = Array.from({length: count}, (_, piece) => {
      const start = offsets[piece];
      const end = offsets[piece + 1];
      assert.ok(end >= start + 2 && end <= vertexCount);
      return pairs(coordinates.slice(start * 2, end * 2));
    });
    return {paths, status: {count, vertexCount, overflow, required, uncertain, candidateOverflow}};
  }
  const inside = readClip();
  const outside = readClip();
  assert.equal(cursor, summaryWordCount);
  return {method, corners, sizes, longestLine, diameter, hullSize, minimumOverflow, subdivisionPolygons, subdivisionStatus, inside, outside};
}

await mkdir(outputDirectory, {recursive: true});
const errors = [];
const cases = [];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
let page;
try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}, reducedMotion: 'reduce'});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  await page.addInitScript(expectedWordCount => {
    const original = GPUBuffer.prototype.getMappedRange;
    GPUBuffer.prototype.getMappedRange = function (...arguments_) {
      const bytes = original.apply(this, arguments_);
      if (bytes.byteLength === expectedWordCount * 4) {
        globalThis.__routeWindowingReadback = Array.from(new Uint32Array(bytes));
        globalThis.__routeWindowingSequence = (globalThis.__routeWindowingSequence ?? 0) + 1;
      }
      return bytes;
    };
  }, summaryWordCount);
  const url = new URL(baseUrl);
  url.hash = '#/story/route-windowing';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await page.waitForFunction(() => globalThis.__routeWindowingSequence > 0, null, {timeout: 90_000});
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const method of ['convex', 'rectilinear']) {
    const control = page.locator('#all-option-method');
    if (await control.inputValue() !== method) {
      const sequence = await page.evaluate(() => globalThis.__routeWindowingSequence);
      await control.selectOption(method);
      await page.waitForFunction(previous => globalThis.__routeWindowingSequence > previous, sequence, {timeout: 90_000});
    }
    cases.push(parseResult(await page.evaluate(() => globalThis.__routeWindowingReadback), method));
    await page.locator('#all-option-view').selectOption('subdivision');
    await page.locator('#all-option-focus').selectOption('loop');
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await page.screenshot({path: `${outputDirectory}/${method}.png`});
  }
  assert.deepEqual(errors, []);
} catch (error) {
  await writeFile(`${outputDirectory}/failure.txt`, String(error.stack ?? error));
  await page?.screenshot({path: `${outputDirectory}/failure.png`}).catch(() => {});
  throw error;
} finally {
  const lastReadback = await page?.evaluate(() => globalThis.__routeWindowingReadback ?? null).catch(() => null);
  await writeFile(`${outputDirectory}/last-readback.json`, JSON.stringify(lastReadback ?? null) + '\n');
  await browser.close();
  await writeFile(`${outputDirectory}/gpu-output.json`, JSON.stringify(cases, null, 2) + '\n');
  await writeFile(`${outputDirectory}/browser-errors.json`, JSON.stringify(errors, null, 2) + '\n');
}
const result = spawnSync(python, [referenceScript, `${outputDirectory}/cpu-reference.json`, `${outputDirectory}/gpu-output.json`], {encoding: 'utf8'});
await writeFile(`${outputDirectory}/geometry-checks.log`, result.stdout + result.stderr);
assert.equal(result.status, 0, result.stdout + result.stderr);
console.log(`Verified ${cases.length} route-windowing partitions and both clip modes against independent GEOS geometry: ${outputDirectory}`);
