// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Verifies live operation runs (src/gpu/runs/entries/<group>/<entry id>.ts) on real WebGPU.
//
// Usage (from the website directory):
//   node scripts/verify-runs.mjs                              all runs
//   node scripts/verify-runs.mjs --group spatial-statistics   every run of one group
//   node scripts/verify-runs.mjs local-morans-i raster-arithmetic   specific entry ids
// Options:
//   --port 3301        port of the harness server; started (and stopped) here when not running
//   --themes dark,light   themes to render (the checks run on the first; others must not be blank)
//   --out <dir>        report folder (default: website/.reports/runs)
//   --headed           show the browser
// Full-size screenshots are saved to <out>/<group>/<entry id>-<theme>.png.
//
// Per run it checks: starts on WebGPU; no GPU validation errors or console errors; canvas not blank;
// the picture changes when parameters move (reports inert sliders); and the picture differs from
// an inputs-only render (the operation's output is what is drawn). It writes <out>/<group>.json and
// a contact sheet <out>/<group>.png. Exit code 1 when any run fails.

import {spawn} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';

const websiteDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const entriesDirectory = path.join(websiteDirectory, 'src/gpu/runs/entries');
const DEFAULT_OUT = path.join(websiteDirectory, '.reports/runs');
const CHANGE_THRESHOLD = 0.002; // fraction of pixels that must differ
// Pointer gesture: positions across the canvas (0 to 1, y down) the verifier visits.
const POINTER_POSITIONS = [
  [0.25, 0.3],
  [0.75, 0.3],
  [0.5, 0.5],
  [0.25, 0.72],
  [0.75, 0.72]
];
// A moving glyph alone changes about 0.1 to 0.3% of pixels; the picture must change more than that.
const POINTER_CHANGE_THRESHOLD = 0.004;
const BLANK_THRESHOLD = 0.01; // fraction of pixels that must differ from the background

function parseArguments(argv) {
  const options = {port: 3301, themes: ['dark'], out: DEFAULT_OUT, group: null, ids: [], headed: false};
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === '--port') options.port = Number(argv[++index]);
    else if (argument === '--group') options.group = argv[++index];
    else if (argument === '--themes') options.themes = argv[++index].split(',');
    else if (argument === '--out') options.out = argv[++index];
    else if (argument === '--headed') options.headed = true;
    else options.ids.push(argument);
  }
  return options;
}

/** Every run on disk as {group, id}; parameter files and non-scene files are skipped. */
function listRuns() {
  const runs = [];
  for (const group of fs.existsSync(entriesDirectory) ? fs.readdirSync(entriesDirectory) : []) {
    const groupDirectory = path.join(entriesDirectory, group);
    if (!fs.statSync(groupDirectory).isDirectory()) continue;
    for (const file of fs.readdirSync(groupDirectory)) {
      if (file.endsWith('.ts') && !file.endsWith('.parameters.ts')) {
        runs.push({group, id: file.slice(0, -3)});
      }
    }
  }
  return runs;
}

async function isServerUp(port) {
  try {
    const response = await fetch(`http://localhost:${port}/run-harness.html`);
    return response.ok;
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

// Runs inside the browser: compares two PNG data URLs and measures one.
const IMAGE_TOOLS = `
window.decodeImage = async dataUrl => {
  const image = new Image();
  image.src = dataUrl;
  await image.decode();
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext('2d', {willReadFrequently: true});
  context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, image.width, image.height).data;
};
window.compareImages = async (firstUrl, secondUrl) => {
  const first = await window.decodeImage(firstUrl);
  const second = await window.decodeImage(secondUrl);
  let different = 0;
  for (let offset = 0; offset < first.length; offset += 4) {
    const delta = Math.abs(first[offset] - second[offset]) + Math.abs(first[offset + 1] - second[offset + 1]) + Math.abs(first[offset + 2] - second[offset + 2]);
    if (delta > 24) different++;
  }
  return different / (first.length / 4);
};
window.measureImage = async dataUrl => {
  const pixels = await window.decodeImage(dataUrl);
  const background = [pixels[0], pixels[1], pixels[2]];
  let different = 0;
  const colors = new Set();
  for (let offset = 0; offset < pixels.length; offset += 4) {
    const delta = Math.abs(pixels[offset] - background[0]) + Math.abs(pixels[offset + 1] - background[1]) + Math.abs(pixels[offset + 2] - background[2]);
    if (delta > 24) different++;
    colors.add(((pixels[offset] >> 4) << 8) | ((pixels[offset + 1] >> 4) << 4) | (pixels[offset + 2] >> 4));
  }
  return {nonBackground: different / (pixels.length / 4), colors: colors.size};
};
`;

async function shoot(page) {
  const buffer = await page.locator('canvas').screenshot();
  return {buffer, url: `data:image/png;base64,${buffer.toString('base64')}`};
}

async function verifyRun(browser, options, {group, id}, theme) {
  const result = {id, group, theme, passed: false, failures: [], warnings: []};
  const page = await browser.newPage({viewport: {width: 960, height: 540}});
  const started = Date.now();
  try {
    // An init script survives the reload Vite triggers when a source file changes mid-run.
    await page.addInitScript(IMAGE_TOOLS);
    await page.goto(`http://localhost:${options.port}/run-harness.html?id=${id}&theme=${theme}`);
    await page.waitForFunction(
      () => window.runHarness && window.runHarness.state.status !== 'loading',
      null,
      {timeout: 90000}
    );
    const state = await page.evaluate(() => JSON.parse(JSON.stringify(window.runHarness.state)));
    result.startMilliseconds = Date.now() - started;
    result.backend = state.backend;
    result.layers = state.layerKinds;
    result.parameters = state.parameters.map(({id: parameterId}) => parameterId);
    if (state.status !== 'ready') {
      result.failures.push(`did not start: ${state.error}`);
      return result;
    }
    if (state.backend !== 'webgpu') result.failures.push(`backend is ${state.backend}, not webgpu`);

    const base = await shoot(page);
    result.screenshot = base.buffer;
    const measured = await page.evaluate(url => window.measureImage(url), base.url);
    result.nonBackground = Number(measured.nonBackground.toFixed(4));
    result.colors = measured.colors;
    if (measured.nonBackground < BLANK_THRESHOLD) result.failures.push('canvas is blank');

    if (theme === options.themes[0]) {
      // Parameters: move each to its extremes; at least one must change the picture.
      result.sliders = [];
      for (const parameter of state.parameters.slice(0, 5)) {
        const effects = [];
        for (const value of [parameter.min, parameter.max]) {
          if (value === parameter.default) continue;
          await page.evaluate(values => window.runHarness.setParameters(values), {[parameter.id]: value});
          const moved = await shoot(page);
          effects.push(await page.evaluate(([a, b]) => window.compareImages(a, b), [base.url, moved.url]));
        }
        await page.evaluate(values => window.runHarness.setParameters(values), {[parameter.id]: parameter.default});
        const change = Math.max(0, ...effects);
        result.sliders.push({id: parameter.id, change: Number(change.toFixed(4))});
        if (change < CHANGE_THRESHOLD) result.warnings.push(`slider "${parameter.id}" does not change the picture`);
      }
      if (result.sliders.length > 0 && result.sliders.every(slider => slider.change < CHANGE_THRESHOLD)) {
        result.failures.push('no slider changes the picture');
      }
      const restored = await shoot(page);
      result.restoreDifference = Number(
        (await page.evaluate(([a, b]) => window.compareImages(a, b), [base.url, restored.url])).toFixed(4)
      );
      if (result.restoreDifference > 0.01) result.warnings.push('picture differs after parameters return to defaults');

      // Output must be drawn: the picture differs from the inputs-only render.
      await page.evaluate(() => window.runHarness.setInputsOnly(true));
      const inputsOnly = await shoot(page);
      result.outputShare = Number(
        (await page.evaluate(([a, b]) => window.compareImages(a, b), [base.url, inputsOnly.url])).toFixed(4)
      );
      if (result.outputShare < 0.01) result.failures.push('picture equals the inputs-only render: output is not drawn');
      await page.evaluate(() => window.runHarness.setInputsOnly(false));
      await page.evaluate(() => window.runHarness.settle());
    }

    if (theme === options.themes[0]) {
      if (!state.gesture) {
        result.warnings.push('no interaction declared: the card falls back to the generic x scrub');
      } else {
        // Pointer gesture: visit five positions; the picture must follow, without recompiling.
        const shots = [];
        const readouts = [];
        for (const position of POINTER_POSITIONS) {
          await page.evaluate(point => window.runHarness.setPointer(point), position);
          shots.push(await shoot(page));
          readouts.push(await page.evaluate(() => window.runHarness.state.readout));
        }
        const compare = (first, second) =>
          page.evaluate(([a, b]) => window.compareImages(a, b), [first.url, second.url]);
        const changes = [];
        for (const shot of shots) changes.push(Number((await compare(base, shot)).toFixed(4)));
        const kept = [];
        for (const shot of shots) {
          let isDistinct = true;
          for (const other of kept) {
            if ((await compare(other, shot)) < POINTER_CHANGE_THRESHOLD) isDistinct = false;
          }
          if (isDistinct) kept.push(shot);
        }
        const pointerState = await page.evaluate(() => JSON.parse(JSON.stringify(window.runHarness.state)));
        await page.evaluate(() => window.runHarness.resetPointer());
        const rested = await shoot(page);
        result.pointer = {
          kind: state.gesture.kind,
          hint: state.gesture.hint,
          changes,
          distinct: kept.length,
          xEffect: Number((await compare(shots[0], shots[1])).toFixed(4)),
          yEffect: Number((await compare(shots[0], shots[3])).toFixed(4)),
          restDifference: Number((await compare(base, rested)).toFixed(4)),
          pointerRebuilds: pointerState.pointerRebuildCount,
          parameters: pointerState.pointerParameterIds,
          readouts
        };
        for (const parameter of state.parameters) {
          if (pointerState.pointerParameterIds.includes(parameter.id) && parameter.label.startsWith('Data: ')) {
            result.warnings.push(`pointer gesture drives the input-data knob "${parameter.id}"`);
          }
        }
        result.pointerShots = [shots[0].buffer, shots[2].buffer, shots[4].buffer];
        if (Math.max(...changes, result.pointer.xEffect, result.pointer.yEffect) < POINTER_CHANGE_THRESHOLD) {
          result.failures.push('pointer gesture does not change the picture');
        } else if (kept.length < 3) {
          result.warnings.push(`pointer gesture gives only ${kept.length} distinct pictures at 5 positions`);
        }
        if (pointerState.pointerRebuildCount > 0) {
          result.failures.push('pointer gesture recompiles the graph: return compile-time parameters in `settle`, not `values`');
        }
        if (result.pointer.restDifference > 0.01) result.warnings.push('picture differs after the pointer returns to rest');
        if (readouts.every(readout => !readout)) result.warnings.push('pointer gesture has no readout');
        for (const parameter of state.parameters) {
          const rest = state.restValues[parameter.id];
          if (rest !== undefined && Math.abs(rest - parameter.default) > parameter.step * 1.5) {
            result.warnings.push(`rest position gives ${parameter.id} ${rest}, not its default ${parameter.default}`);
          }
        }
      }
    }

    const final = await page.evaluate(() => JSON.parse(JSON.stringify(window.runHarness.state)));
    result.runCount = final.runCount;
    result.rebuildCount = final.rebuildCount;
    for (const message of final.gpuErrors) result.failures.push(`GPU error: ${message}`);
    for (const message of final.consoleErrors) {
      if (!/favicon/.test(message)) result.failures.push(`console error: ${message}`);
    }
    result.passed = result.failures.length === 0;
  } catch (error) {
    result.failures.push(`harness error: ${String(error).slice(0, 300)}`);
  } finally {
    await page.close();
  }
  return result;
}

async function writeContactSheet(browser, results, file) {
  const cells = results
    .filter(result => result.screenshot)
    .map(
      result => `<figure><img src="data:image/png;base64,${result.screenshot.toString('base64')}"><figcaption style="color:${result.passed ? '#7ddc8a' : '#ff6b6b'}">${result.id} ${result.passed ? '' : '(FAILED)'}<br><small>${(result.layers ?? []).join(' + ')}</small></figcaption></figure>`
    )
    .join('');
  const page = await browser.newPage({viewport: {width: 1300, height: 400}});
  await page.setContent(
    `<body style="margin:0;background:#101521;font:12px monospace;color:#ccd"><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:8px">${cells}</div><style>figure{margin:0}img{width:100%;display:block}</style></body>`
  );
  await page.screenshot({path: file, fullPage: true});
  await page.close();
}

/**
 * One row per run: the picture at three pointer positions, with the gesture and readouts. Written
 * as pages of eight runs (<name>-1.png, <name>-2.png, ...) so each page stays legible when read.
 */
async function writePointerSheet(browser, results, file) {
  const withShots = results.filter(result => result.pointerShots);
  const PAGE_SIZE = 8;
  for (let start = 0; start < withShots.length; start += PAGE_SIZE) {
    const rows = withShots
      .slice(start, start + PAGE_SIZE)
      .map(result => {
        const images = result.pointerShots
          .map(shot => `<img src="data:image/png;base64,${shot.toString('base64')}">`)
          .join('');
        const readouts = [0, 2, 4].map(index => result.pointer.readouts[index] ?? '(no readout)').join(' | ');
        return `<section><h2>${result.id} <small>${result.pointer.kind}: ${result.pointer.hint} · distinct ${result.pointer.distinct}/5 · x ${(result.pointer.xEffect * 100).toFixed(1)}% y ${(result.pointer.yEffect * 100).toFixed(1)}%</small></h2><div>${images}</div><p>${readouts}</p></section>`;
      })
      .join('');
    const page = await browser.newPage({viewport: {width: 1300, height: 400}});
    await page.setContent(
      `<body style="margin:0;background:#101521;font:12px monospace;color:#ccd;padding:8px">${rows}<style>section{margin-bottom:10px}h2{font-size:13px;margin:0 0 3px}small{color:#8a93a8;font-weight:normal}div{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}img{width:100%;display:block}p{margin:2px 0 0;color:#8a93a8}</style></body>`
    );
    await page.screenshot({path: file.replace(/\.png$/, `-${start / PAGE_SIZE + 1}.png`), fullPage: true});
    await page.close();
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const all = listRuns();
  let selected = all;
  if (options.group) selected = all.filter(run => run.group === options.group);
  if (options.ids.length > 0) selected = all.filter(run => options.ids.includes(run.id));
  if (selected.length === 0) {
    console.error('No runs match. Runs live in src/gpu/runs/entries/<group>/<entry id>.ts');
    process.exit(2);
  }
  fs.mkdirSync(options.out, {recursive: true});
  const server = await startServer(options.port);
  const browser = await chromium.launch({
    headless: !options.headed,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-features=Vulkan,WebGPUService', '--use-angle=metal']
  });
  const byGroup = new Map();
  for (const run of selected) {
    const results = [];
    for (const theme of options.themes) {
      let result = await verifyRun(browser, options, run, theme);
      // Vite reloads the page when a source file changes under it (every `.parameters.ts` is in
      // the harness's module graph) or when it re-optimizes dependencies: retry.
      for (
        let attempt = 0;
        attempt < 4 && result.failures.some(failure => failure.startsWith('harness error'));
        attempt++
      ) {
        result = await verifyRun(browser, options, run, theme);
      }
      results.push(result);
      if (result.screenshot) {
        const shotDirectory = path.join(options.out, run.group);
        fs.mkdirSync(shotDirectory, {recursive: true});
        fs.writeFileSync(path.join(shotDirectory, `${run.id}-${theme}.png`), result.screenshot);
        (result.pointerShots ?? []).forEach((shot, index) => {
          fs.writeFileSync(path.join(shotDirectory, `${run.id}-pointer-${index}.png`), shot);
        });
      }
      const flag = result.passed ? 'PASS' : 'FAIL';
      console.log(`${flag} ${run.group}/${run.id} [${theme}] ${result.startMilliseconds ?? '-'}ms layers=${(result.layers ?? []).join('+')}${result.failures.length ? '\n   ' + result.failures.join('\n   ') : ''}${result.warnings.length ? '\n   warn: ' + result.warnings.join('; ') : ''}`);
    }
    byGroup.set(run.group, [...(byGroup.get(run.group) ?? []), ...results]);
  }
  let failed = 0;
  for (const [group, results] of byGroup) {
    const primary = results.filter(result => result.theme === options.themes[0]);
    await writeContactSheet(browser, primary, path.join(options.out, `${group}.png`));
    await writePointerSheet(browser, primary, path.join(options.out, `${group}-pointer.png`));
    const serializable = results.map(({screenshot, pointerShots, ...rest}) => rest);
    fs.writeFileSync(
      path.join(options.out, `${group}.json`),
      JSON.stringify({group, total: results.length, passed: results.filter(r => r.passed).length, results: serializable}, null, 2)
    );
    failed += results.filter(result => !result.passed).length;
  }
  console.log(`\n${selected.length * options.themes.length - failed} / ${selected.length * options.themes.length} passed. Reports in ${options.out}`);
  await browser.close();
  server?.kill();
  process.exit(failed > 0 ? 1 : 0);
}

main();
