// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Verifies the pointer gesture of the flagship demos (src/gpu/demos) on real WebGPU.
//
// Usage (from the website directory):
//   node scripts/verify-demos.mjs                 every demo
//   node scripts/verify-demos.mjs idw dbscan      specific demo ids (file names in src/gpu/demos)
// Options: --port 3301 (the harness server of verify-runs.mjs; must be running, or start it with
//   `node scripts/verify-runs.mjs <any id>` once), --out <dir> (default website/.reports/demos)
//
// Per demo it loads the scene live, visits five pointer positions and checks that the scene
// declares a gesture, that the picture follows the pointer and that a readout is reported. It
// writes <out>/demos.json and a sheet <out>/demos-pointer.png with three positions per demo.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';

const websiteDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// Demo id (file name) and one catalog entry id that shows it.
const DEMOS = {
  'kernel-density': 'kernel-density',
  'local-morans-i': 'local-morans-i',
  dbscan: 'dbscan-clustering',
  'k-means': 'k-means-clustering',
  idw: 'inverse-distance-weighting',
  'delaunay-voronoi': 'delaunay-tessellation',
  'terrain-hydrology': 'terrain-flow',
  'network-isochrones': 'network-reachability',
  'hot-spots': 'hot-spot-analysis',
  'shortest-path': 'shortest-path'
};
const POSITIONS = [
  [0.25, 0.3],
  [0.75, 0.3],
  [0.5, 0.5],
  [0.25, 0.72],
  [0.75, 0.72]
];
const CHANGE_THRESHOLD = 0.004;

const argv = process.argv.slice(2);
let port = 3301;
let out = path.join(websiteDirectory, '.reports/demos');
const ids = [];
for (let index = 0; index < argv.length; index++) {
  if (argv[index] === '--port') port = Number(argv[++index]);
  else if (argv[index] === '--out') out = argv[++index];
  else ids.push(argv[index]);
}
const selected = Object.keys(DEMOS).filter(id => ids.length === 0 || ids.includes(id));
fs.mkdirSync(out, {recursive: true});

const COMPARE = `
window.compareImages = async (firstUrl, secondUrl) => {
  const decode = async url => {
    const image = new Image();
    image.src = url;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext('2d', {willReadFrequently: true});
    context.drawImage(image, 0, 0);
    return context.getImageData(0, 0, image.width, image.height).data;
  };
  const first = await decode(firstUrl);
  const second = await decode(secondUrl);
  let different = 0;
  for (let offset = 0; offset < first.length; offset += 4) {
    const delta = Math.abs(first[offset] - second[offset]) + Math.abs(first[offset + 1] - second[offset + 1]) + Math.abs(first[offset + 2] - second[offset + 2]);
    if (delta > 24) different++;
  }
  return different / (first.length / 4);
};
`;

const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-features=Vulkan,WebGPUService', '--use-angle=metal']
});
const results = [];
async function verifyDemo(id) {
  const result = {id, passed: false, failures: [], warnings: []};
  const page = await browser.newPage({viewport: {width: 960, height: 540}});
  try {
    // An init script survives the reload Vite triggers when a source file changes mid-run.
    await page.addInitScript(COMPARE);
    await page.goto(`http://localhost:${port}/run-harness.html?id=${DEMOS[id]}&demo=1&theme=dark`);
    await page.waitForFunction(() => window.runHarness && window.runHarness.state.status !== 'loading', null, {timeout: 90000});
    const state = await page.evaluate(() => JSON.parse(JSON.stringify(window.runHarness.state)));
    if (state.status !== 'ready') {
      result.failures.push(`did not start: ${state.error}`);
    } else {
      const shoot = async () => {
        const buffer = await page.locator('canvas').screenshot();
        return {buffer, url: `data:image/png;base64,${buffer.toString('base64')}`};
      };
      const compare = (first, second) => page.evaluate(([a, b]) => window.compareImages(a, b), [first.url, second.url]);
      await page.evaluate(() => window.runHarness.settle());
      result.gesture = await page.evaluate(() => window.runHarness.state.gesture);
      if (!result.gesture) result.failures.push('demo declares no gesture');
      const shots = [];
      const readouts = [];
      for (const position of POSITIONS) {
        await page.evaluate(point => window.runHarness.setPointer(point), position);
        shots.push(await shoot());
        readouts.push(await page.evaluate(() => window.runHarness.state.readout));
      }
      const kept = [];
      for (const shot of shots) {
        let isDistinct = true;
        for (const other of kept) if ((await compare(other, shot)) < CHANGE_THRESHOLD) isDistinct = false;
        if (isDistinct) kept.push(shot);
      }
      result.distinct = kept.length;
      result.xEffect = Number((await compare(shots[0], shots[1])).toFixed(4));
      result.yEffect = Number((await compare(shots[0], shots[3])).toFixed(4));
      result.readouts = readouts;
      result.shots = [shots[0].buffer, shots[2].buffer, shots[4].buffer];
      if (Math.max(result.xEffect, result.yEffect) < CHANGE_THRESHOLD && kept.length < 2) {
        result.failures.push('pointer gesture does not change the picture');
      } else if (kept.length < 3) {
        result.warnings.push(`only ${kept.length} distinct pictures at 5 pointer positions`);
      }
      if (readouts.every(readout => !readout)) result.warnings.push('no readout');
      const final = await page.evaluate(() => JSON.parse(JSON.stringify(window.runHarness.state)));
      for (const message of final.gpuErrors) result.failures.push(`GPU error: ${message}`);
      for (const message of final.consoleErrors) if (!/favicon/.test(message)) result.failures.push(`console error: ${message}`);
    }
    result.passed = result.failures.length === 0;
  } catch (error) {
    result.failures.push(`harness error: ${String(error).slice(0, 300)}`);
  } finally {
    await page.close();
  }
  return result;
}
for (const id of selected) {
  let result = await verifyDemo(id);
  // Vite reloads the page when a source file changes under it: retry.
  for (
    let attempt = 0;
    attempt < 4 && result.failures.some(failure => failure.startsWith('harness error'));
    attempt++
  ) {
    result = await verifyDemo(id);
  }
  results.push(result);
  console.log(`${result.passed ? 'PASS' : 'FAIL'} ${id} ${result.gesture ? result.gesture.kind + ': ' + result.gesture.hint : ''} distinct=${result.distinct ?? '-'}${result.failures.length ? '\n   ' + result.failures.join('\n   ') : ''}${result.warnings.length ? '\n   warn: ' + result.warnings.join('; ') : ''}`);
}
const rows = results
  .filter(result => result.shots)
  .map(result => `<section><h2>${result.id} <small>${result.gesture ? result.gesture.kind + ': ' + result.gesture.hint : 'no gesture'} · distinct ${result.distinct}/5 · x ${(result.xEffect * 100).toFixed(1)}% y ${(result.yEffect * 100).toFixed(1)}%</small></h2><div>${result.shots.map(shot => `<img src="data:image/png;base64,${shot.toString('base64')}">`).join('')}</div><p>${[0, 2, 4].map(index => result.readouts[index] ?? '(no readout)').join(' | ')}</p></section>`)
  .join('');
if (rows) {
  const page = await browser.newPage({viewport: {width: 1300, height: 400}});
  await page.setContent(`<body style="margin:0;background:#101521;font:12px monospace;color:#ccd;padding:8px">${rows}<style>section{margin-bottom:10px}h2{font-size:13px;margin:0 0 3px}small{color:#8a93a8;font-weight:normal}div{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}img{width:100%;display:block}p{margin:2px 0 0;color:#8a93a8}</style></body>`);
  await page.screenshot({path: path.join(out, 'demos-pointer.png'), fullPage: true});
  await page.close();
}
fs.writeFileSync(path.join(out, 'demos.json'), JSON.stringify(results.map(({shots, ...rest}) => rest), null, 2));
await browser.close();
const failed = results.filter(result => !result.passed).length;
console.log(`\n${results.length - failed} / ${results.length} passed. Reports in ${out}`);
process.exit(failed > 0 ? 1 : 0);
