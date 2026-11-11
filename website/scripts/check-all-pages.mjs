// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Opens every built page (dist/client, served statically) in headless Chromium with real WebGPU at
// 1280 x 800 and checks that each live canvas starts, nothing logs an error, and that moving a
// slider changes the picture. Covers all /fn/<id> pages, the home page and the how-it-works pages.
//
// Usage (from the website folder, after `yarn website-build`):
//   node scripts/check-all-pages.mjs                  all pages
//   node scripts/check-all-pages.mjs <entry id>...    specific function pages
// Options: --port 4177, --workers 3, --out <json file>, --only fn|home|how,
//   --base /luma-contributors (serve the build under the sub-path it was built for with
//   SITE_BASE_PATH; requests outside it get 404, as on GitHub Pages)
// Exit code 1 when any page fails.

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';

const websiteDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const distDirectory = path.join(websiteDirectory, 'dist/client');
const DEFAULT_OUT = path.join(websiteDirectory, '.reports/pages/all-pages.json');
const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.woff2': 'font/woff2',
  '.ico': 'image/x-icon'
};

function parseArguments(argv) {
  const options = {port: 4177, workers: 3, out: DEFAULT_OUT, only: null, base: '', ids: []};
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === '--port') options.port = Number(argv[++index]);
    else if (argument === '--workers') options.workers = Number(argv[++index]);
    else if (argument === '--out') options.out = argv[++index];
    else if (argument === '--only') options.only = argv[++index];
    else if (argument === '--base') options.base = argv[++index].replace(/\/$/, '');
    else options.ids.push(argument);
  }
  return options;
}

function serve(port, base) {
  const server = http.createServer((request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://x').pathname);
    if (base && pathname !== base && !pathname.startsWith(`${base}/`)) {
      response.writeHead(404);
      response.end('not found');
      return;
    }
    let file = path.join(distDirectory, pathname.slice(base.length));
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
    if (!fs.existsSync(file)) {
      response.writeHead(404);
      response.end('not found');
      return;
    }
    response.writeHead(200, {'content-type': MIME[path.extname(file)] ?? 'application/octet-stream'});
    fs.createReadStream(file).pipe(response);
  });
  return new Promise(resolve => server.listen(port, () => resolve(server)));
}

function listRunIds() {
  const entriesDirectory = path.join(websiteDirectory, 'src/gpu/runs/entries');
  const ids = new Set();
  for (const group of fs.readdirSync(entriesDirectory)) {
    const groupDirectory = path.join(entriesDirectory, group);
    if (!fs.statSync(groupDirectory).isDirectory()) continue;
    for (const file of fs.readdirSync(groupDirectory)) {
      if (file.endsWith('.ts') && !file.endsWith('.parameters.ts')) ids.add(file.slice(0, -3));
    }
  }
  return ids;
}

function listFunctionIds() {
  return fs.readdirSync(path.join(distDirectory, 'fn')).filter(name =>
    fs.existsSync(path.join(distDirectory, 'fn', name, 'index.html'))
  );
}

const INIT_SCRIPT = () => {
  window.__gpuErrors = [];
  const request = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (...parameters) {
    const device = await request.apply(this, parameters);
    device.addEventListener('uncapturederror', event =>
      window.__gpuErrors.push(String(event.error.message).slice(0, 300))
    );
    return device;
  };
};

async function shoot(locator) {
  return (await locator.screenshot()).toString('base64');
}

async function comparePictures(page, firstBase64, secondBase64) {
  return page.evaluate(async ([first, second]) => {
    const decode = async data => {
      const image = new Image();
      image.src = `data:image/png;base64,${data}`;
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      const context = canvas.getContext('2d', {willReadFrequently: true});
      context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, image.width, image.height).data;
    };
    const [a, b] = await Promise.all([decode(first), decode(second)]);
    if (a.length !== b.length) return 1;
    let different = 0;
    for (let offset = 0; offset < a.length; offset += 4) {
      const delta = Math.abs(a[offset] - b[offset]) + Math.abs(a[offset + 1] - b[offset + 1]) + Math.abs(a[offset + 2] - b[offset + 2]);
      if (delta > 24) different++;
    }
    return different / (a.length / 4);
  }, [firstBase64, secondBase64]);
}

/** Moves one control to its far end and waits until the picture changes (or gives up). */
async function moveControl(page, canvas, control, before) {
  const handle = await control.elementHandle();
  const isOptions = await handle.evaluate(element => element.tagName === 'BUTTON');
  if (isOptions) await control.click();
  else {
    await handle.evaluate(element => {
      const target = Number(element.value) - Number(element.min) > (Number(element.max) - Number(element.min)) / 2 ? element.min : element.max;
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      setter.call(element, target);
      element.dispatchEvent(new Event('input', {bubbles: true}));
    });
  }
  const deadline = Date.now() + 12000;
  let change = 0;
  while (Date.now() < deadline) {
    await page.waitForTimeout(500);
    change = await comparePictures(page, before, await shoot(canvas));
    if (change > 0.002) break;
  }
  return change;
}

async function checkPage(browser, options, spec) {
  const result = {id: spec.id, url: spec.url, kind: spec.kind, live: false, errors: []};
  const page = await browser.newPage({viewport: {width: 1280, height: 800}});
  await page.addInitScript(INIT_SCRIPT);
  page.on('console', message => {
    if (message.type() === 'error' && !/favicon/.test(message.text())) result.errors.push(`console: ${message.text().slice(0, 200)}`);
  });
  page.on('pageerror', error => result.errors.push(`pageerror: ${String(error).slice(0, 200)}`));
  try {
    await page.goto(`http://localhost:${options.port}${options.base}${spec.url}`, {waitUntil: 'load'});
    const selector = spec.kind === 'fn' ? '.function-visual .gpu-canvas' : '.gpu-canvas';
    const canvases = page.locator(selector);
    const count = await canvases.count();
    result.canvasCount = count;
    if (spec.kind === 'fn') {
      const text = await page.locator('.function-visual').innerText();
      result.illustrative = /Illustrative scene/.test(text);
    }
    // Scroll each canvas into view in turn so it starts, then wait for it to be ready.
    let readyCount = 0;
    for (let index = 0; index < count; index++) {
      const canvas = canvases.nth(index);
      await canvas.scrollIntoViewIfNeeded();
      try {
        await page.waitForFunction(
          ([selectorText, position]) => document.querySelectorAll(selectorText)[position]?.dataset.ready === 'true',
          [selector, index],
          {timeout: 45000}
        );
        readyCount++;
      } catch {
        result.errors.push(`canvas ${index} never reached data-ready=true`);
      }
    }
    // Pages without a live canvas (pure diagrams) pass; a function page must have one.
    result.live = count > 0 ? readyCount === count : spec.kind !== 'fn';
    result.readyCount = readyCount;
    if (spec.kind === 'fn' && result.live) {
      const canvas = canvases.first();
      await canvas.scrollIntoViewIfNeeded();
      const controls = page.locator('.function-visual .gpu-controls-sliders input[type=range], .function-visual .gpu-controls-sliders .gpu-segments button:not([aria-pressed=true])');
      const controlCount = await controls.count();
      result.sliderCount = await page.locator('.function-visual .gpu-control').count();
      if (controlCount > 0) {
        await page.waitForTimeout(600);
        const before = await shoot(canvas);
        result.firstControlChange = Number((await moveControl(page, canvas, controls.first(), before)).toFixed(4));
        result.sliderChanges = result.firstControlChange > 0.002;
        if (!result.sliderChanges) {
          // The first control may be a subtle prop: report whether any of the others moves the picture.
          result.anyControlChange = false;
          for (let index = 1; index < Math.min(controlCount, 6); index++) {
            const baseline = await shoot(canvas);
            const change = await moveControl(page, canvas, controls.nth(index), baseline);
            if (change > 0.002) {
              result.anyControlChange = true;
              break;
            }
          }
        }
      }
    }
    const gpuErrors = await page.evaluate(() => window.__gpuErrors ?? []);
    for (const message of gpuErrors) result.errors.push(`gpu: ${message}`);
  } catch (error) {
    result.errors.push(`harness: ${String(error).slice(0, 200)}`);
  } finally {
    await page.close();
  }
  result.passed =
    result.live &&
    result.errors.length === 0 &&
    (result.sliderChanges === undefined || result.sliderChanges || result.anyControlChange === true);
  return result;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const runIds = listRunIds();
  const specs = [];
  const functionIds = options.ids.length > 0 ? options.ids : listFunctionIds();
  if (!options.only || options.only === 'fn') {
    for (const id of functionIds) specs.push({id, url: `/fn/${id}`, kind: 'fn', hasRun: runIds.has(id)});
  }
  if (options.ids.length === 0 && (!options.only || options.only === 'home')) specs.push({id: 'home', url: '/', kind: 'home'});
  if (options.ids.length === 0 && (!options.only || options.only === 'how')) {
    specs.push({id: 'how-it-works', url: '/how-it-works', kind: 'how'});
    for (const topic of fs.readdirSync(path.join(distDirectory, 'how-it-works'))) {
      if (fs.existsSync(path.join(distDirectory, 'how-it-works', topic, 'index.html'))) {
        specs.push({id: `how-it-works/${topic}`, url: `/how-it-works/${topic}`, kind: 'how'});
      }
    }
  }
  const server = await serve(options.port, options.base);
  const results = [];
  async function work(browser, queue) {
    for (let spec = queue.shift(); spec; spec = queue.shift()) {
      let result = await checkPage(browser, options, spec);
      if (!result.passed) result = await checkPage(browser, options, spec); // one retry for load flakes
      result.hasRun = spec.hasRun ?? false;
      results.push(result);
      console.log(`${result.passed ? 'PASS' : 'FAIL'} ${spec.url} ${result.errors.slice(0, 2).join(' | ')}${result.sliderChanges === false ? ` (first slider inert; any=${result.anyControlChange})` : ''}`);
    }
  }
  // Each function compiles a new GPU graph. Restart Chromium between batches so
  // driver pipeline caches stay bounded on a full-catalog sweep.
  const batchSize = 40;
  try {
    for (let first = 0; first < specs.length; first += batchSize) {
      const browser = await chromium.launch({
        args: [
          '--enable-unsafe-webgpu',
          '--ignore-gpu-blocklist',
          '--enable-features=Vulkan,WebGPUService',
          ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])
        ]
      });
      try {
        const queue = specs.slice(first, first + batchSize);
        await Promise.all(Array.from({length: options.workers}, () => work(browser, queue)));
      } finally {
        await browser.close();
      }
    }
  } finally {
    server.close();
  }
  results.sort((a, b) => a.url.localeCompare(b.url));
  const functionResults = results.filter(result => result.kind === 'fn');
  const summary = {
    pages: results.length,
    passed: results.filter(result => result.passed).length,
    functionPages: functionResults.length,
    functionLive: functionResults.filter(result => result.live).length,
    withRun: functionResults.filter(result => result.hasRun).length,
    illustrative: functionResults.filter(result => result.illustrative).length,
    demoOnly: functionResults.filter(result => !result.hasRun && !result.illustrative).length,
    withSliders: functionResults.filter(result => result.sliderCount > 0).length,
    firstSliderChanges: functionResults.filter(result => result.sliderChanges).length,
    firstSliderInertButAnyChanges: functionResults.filter(result => result.sliderChanges === false && result.anyControlChange).length,
    failed: results.filter(result => !result.passed).map(result => result.url)
  };
  fs.mkdirSync(path.dirname(options.out), {recursive: true});
  fs.writeFileSync(options.out, JSON.stringify({summary, results}, null, 2));
  console.log(JSON.stringify(summary, null, 2));
  process.exit(summary.failed.length > 0 ? 1 : 0);
}

main();
