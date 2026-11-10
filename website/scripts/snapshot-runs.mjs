// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Renders every catalog entry's live run or flagship demo on WebGPU in both themes and
// saves a small WebP of the output to public/runs/<entry id>-<theme>.webp. The pictures are the
// static fallback of the function pages and cards, so prerendered HTML already shows the
// operation's own output before the live canvas starts.
//
// Usage (from the website folder):
//   node scripts/snapshot-runs.mjs                 all catalog entries
//   node scripts/snapshot-runs.mjs <entry id>...   specific entries
// Options: --port 3302 (harness server, started here when not running), --workers 3,
//   --quality 0.62, --width 640

import {spawn} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
import {getEntrySeed} from '../src/catalog/entry-seed.ts';

const websiteDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const entriesDirectory = path.join(websiteDirectory, 'src/catalog/entries');
const outputDirectory = path.join(websiteDirectory, 'public/runs');
const THEMES = ['dark', 'light'];

function parseArguments(argv) {
  const options = {port: 3302, workers: 3, quality: 0.62, width: 640, ids: []};
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === '--port') options.port = Number(argv[++index]);
    else if (argument === '--workers') options.workers = Number(argv[++index]);
    else if (argument === '--quality') options.quality = Number(argv[++index]);
    else if (argument === '--width') options.width = Number(argv[++index]);
    else options.ids.push(argument);
  }
  return options;
}

/** Include flagship demos, which sit outside the run registry. Use each card's actual input seed. */
async function loadCatalogSeeds() {
  const seeds = new Map();
  for (const file of fs.readdirSync(entriesDirectory).filter(name => name.endsWith('.ts'))) {
    const {entries} = await import(new URL(`../src/catalog/entries/${file}`, import.meta.url));
    for (const entry of entries) seeds.set(entry.id, getEntrySeed(entry));
  }
  return seeds;
}

async function isServerUp(port) {
  try {
    return (await fetch(`http://localhost:${port}/run-harness.html`)).ok;
  } catch {
    return false;
  }
}

async function startServer(port) {
  if (await isServerUp(port)) return null;
  const child = spawn(
    path.join(websiteDirectory, '../node_modules/.bin/vite'),
    ['dev', '--config', 'scripts/harness.vite.config.ts', '--port', String(port), '--strictPort'],
    {cwd: websiteDirectory, stdio: 'ignore'}
  );
  for (let attempt = 0; attempt < 60; attempt++) {
    if (await isServerUp(port)) return child;
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  child.kill();
  throw new Error(`Harness server did not start on port ${port}`);
}

/** Converts a PNG buffer to a scaled WebP buffer using the browser's own encoder. */
async function encodeWebp(page, png, options) {
  const base64 = await page.evaluate(
    async ([data, width, quality]) => {
      const image = new Image();
      image.src = `data:image/png;base64,${data}`;
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = Math.round((width * image.height) / image.width);
      const context = canvas.getContext('2d');
      context.imageSmoothingQuality = 'high';
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      // Reject empty captures before replacing a working poster. Allow sparse line drawings.
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      let visible = 0;
      for (let index = 0; index < pixels.length; index += 4) {
        const difference = Math.abs(pixels[index] - pixels[0])
          + Math.abs(pixels[index + 1] - pixels[1]) + Math.abs(pixels[index + 2] - pixels[2]);
        if (difference > 24) visible++;
      }
      if (visible < canvas.width * canvas.height * 0.002) throw new Error('Blank poster');
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/webp', quality));
      const bytes = new Uint8Array(await blob.arrayBuffer());
      let text = '';
      for (let offset = 0; offset < bytes.length; offset += 8192) {
        text += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
      }
      return btoa(text);
    },
    [png.toString('base64'), options.width, options.quality]
  );
  return Buffer.from(base64, 'base64');
}

async function renderRun(context, converter, options, id, theme, seed) {
  const page = await context.newPage();
  try {
    const query = new URLSearchParams({id, theme, poster: '1', seed: String(seed)});
    await page.goto(`http://localhost:${options.port}/run-harness.html?${query}`);
    await page.waitForFunction(
      () => window.runHarness && window.runHarness.state.status !== 'loading',
      null,
      {timeout: 90000}
    );
    const state = await page.evaluate(() => window.runHarness.state);
    if (state.status !== 'ready') throw new Error(`did not start: ${state.error}`);
    if (state.backend !== 'webgpu') throw new Error(`Expected WebGPU, received ${state.backend}`);
    await page.evaluate(() => window.runHarness.settle());
    const png = await page.locator('canvas').screenshot();
    const errors = await page.evaluate(() => {
      const {gpuErrors, consoleErrors} = window.runHarness.state;
      return [...gpuErrors, ...consoleErrors];
    });
    if (errors.length) throw new Error(errors.join('\n'));
    const webp = await encodeWebp(converter, png, options);
    const destination = path.join(outputDirectory, `${id}-${theme}.webp`);
    const temporary = `${destination}.${process.pid}.tmp`;
    try {
      fs.writeFileSync(temporary, webp);
      fs.renameSync(temporary, destination);
    } finally {
      fs.rmSync(temporary, {force: true});
    }
    return webp.length;
  } finally {
    await page.close();
  }
}

async function capturePosters(browser, options, seeds, ids) {
  const context = await browser.newContext({viewport: {width: 960, height: 540}});
  const converter = await context.newPage();
  const queue = ids.flatMap(id => THEMES.map(theme => ({id, theme})));
  const failures = [];
  let totalBytes = 0;
  let done = 0;
  async function work() {
    for (let job = queue.shift(); job; job = queue.shift()) {
      let lastError = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const bytes = await renderRun(context, converter, options, job.id, job.theme, seeds.get(job.id));
          totalBytes += bytes;
          lastError = null;
          break;
        } catch (error) {
          lastError = error;
        }
      }
      done++;
      if (lastError) failures.push({...job, error: String(lastError).slice(0, 200)});
      if (done % 20 === 0) console.log(`${done}/${ids.length * THEMES.length} (${(totalBytes / 1e6).toFixed(1)} MB)`);
    }
  }
  await Promise.all(Array.from({length: options.workers}, work));
  console.log(`rendered ${done - failures.length}/${done}, ${(totalBytes / 1e6).toFixed(2)} MB`);
  for (const failure of failures) console.log(`FAIL ${failure.id} [${failure.theme}] ${failure.error}`);
  process.exitCode = failures.length > 0 ? 1 : 0;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const seeds = await loadCatalogSeeds();
  const ids = options.ids.length ? options.ids : [...seeds.keys()].sort();
  for (const id of ids) {
    if (!seeds.has(id)) throw new Error(`Unknown catalog entry "${id}"`);
  }
  fs.mkdirSync(outputDirectory, {recursive: true});
  const server = await startServer(options.port);
  let browser;
  try {
    browser = await chromium.launch({
      args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-features=Vulkan,WebGPUService',
        ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])]
    });
    await capturePosters(browser, options, seeds, ids);
  } finally {
    server?.kill();
    await browser?.close();
  }
}

main();
