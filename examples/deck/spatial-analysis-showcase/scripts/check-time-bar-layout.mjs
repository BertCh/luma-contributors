// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {build} from 'esbuild';
import {chromium} from 'playwright';

const workspace = new URL('../', import.meta.url);
const outputFile = process.argv[2] ?? '/tmp/time-bar-layout.json';
const timeBarEntry = new URL('src/shell/time-bar.ts', workspace).pathname;
const resolveDirectory = new URL('src/shell/', workspace).pathname;
const productionCss = await readFile(new URL('src/shell/time-bar.css', workspace), 'utf8');
const fixtureSource = `
import {TimeBar} from './time-bar.ts';
import {OptionsStore} from '../scenes/options-state.ts';
const options = [
  {kind: 'toggle', id: 'play', label: 'Play', group: 'Playback', apply: 'param', default: false},
  {kind: 'slider', id: 'time', label: 'Time', group: 'Playback', apply: 'param', min: 0, max: 7200, step: 10, default: 7200},
  {kind: 'select', id: 'speed', label: 'Speed', group: 'Playback', apply: 'param', default: '120', options: [
    {value: '30', label: '30x (two minutes of timetable per four seconds)'},
    {value: '60', label: '60x'},
    {value: '120', label: '120x (the whole window in one minute)'},
    {value: '300', label: '300x'},
    {value: '600', label: '600x'}
  ]}
];
const store = new OptionsStore(options);
const bar = new TimeBar(store, {
  time: 'time',
  play: 'play',
  speed: 'speed',
  format: value => value === 7200 ? '09:00 CEST' : '07:00 CEST',
  ticks: [{at: 0, label: '07:00'}, {at: 3600, label: '08:00'}, {at: 7200, label: '09:00'}]
});
document.querySelector('#map-slot').append(bar.element);
window.timeBarFixture = {bar, store};
`;
const bundle = await build({
  stdin: {contents: fixtureSource, resolveDir: resolveDirectory, sourcefile: 'time-bar-layout-fixture.ts', loader: 'ts'},
  bundle: true,
  write: false,
  format: 'iife',
  platform: 'browser',
  target: 'es2022',
  loader: {'.css': 'empty'}
});
const browser = await chromium.launch({headless: true, args: ['--disable-gpu']});
const results = [];
let passed = false;
let failureReason = null;
try {
  for (const viewportWidth of [320, 390]) {
    const context = await browser.newContext({
      viewport: {width: viewportWidth, height: 720},
      isMobile: true,
      hasTouch: true
    });
    const page = await context.newPage();
    await page.setContent(`<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>
      html, body { margin: 0; width: 100%; min-height: 100%; }
      body { overflow-x: auto; }
      #map-slot { position: relative; width: 100vw; height: 720px; }
      :root { --map-ink: #17202b; --map-paper: #fff; --map-accent: #e4573d; --map-furniture-bg: #f5f7fa; --map-ink-muted: #68717d; --map-water: #90bed6; }
    </style><style>${productionCss}</style></head><body><main id="map-slot"></main></body></html>`);
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.waitForFunction(() => document.querySelector('.time-bar-speed')?.value === '120');
    const result = await page.evaluate(() => {
      const bar = document.querySelector('.time-bar');
      const speed = document.querySelector('.time-bar-speed');
      const play = document.querySelector('.time-bar-play');
      const time = document.querySelector('.time-bar-value');
      return {
        viewportWidth: window.innerWidth,
        documentWidth: document.documentElement.scrollWidth,
        bodyWidth: document.body.scrollWidth,
        barWidth: bar?.getBoundingClientRect().width ?? 0,
        barScrollWidth: bar?.scrollWidth ?? 0,
        barClientWidth: bar?.clientWidth ?? 0,
        speedWidth: speed?.getBoundingClientRect().width ?? 0,
        speedHeight: speed?.getBoundingClientRect().height ?? 0,
        selectedSpeedLabel: speed?.selectedOptions[0]?.textContent?.trim() ?? '',
        playHeight: play?.getBoundingClientRect().height ?? 0,
        timeLabel: time?.textContent?.trim() ?? '',
        hasWebGpuCanvas: document.querySelector('canvas') !== null
      };
    });
    assert.equal(result.documentWidth, viewportWidth, `${viewportWidth}px: document scrollWidth`);
    assert.equal(result.bodyWidth, viewportWidth, `${viewportWidth}px: body scrollWidth`);
    assert.ok(result.barWidth <= viewportWidth - 16, `${viewportWidth}px: time bar exceeds horizontal inset`);
    assert.ok(result.barScrollWidth <= result.barClientWidth, `${viewportWidth}px: time bar children overflow its content box`);
    assert.equal(result.speedHeight, 44, `${viewportWidth}px: speed touch target height`);
    assert.equal(result.playHeight, 44, `${viewportWidth}px: play touch target height`);
    assert.equal(result.selectedSpeedLabel, '120x (the whole window in one minute)');
    assert.equal(result.timeLabel, '09:00 CEST');
    assert.equal(result.hasWebGpuCanvas, false);
    results.push(result);
    await context.close();
  }
  passed = true;
} catch (error) {
  failureReason = String(error);
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, `${JSON.stringify({passed, status: passed ? 'passed' : 'failed', failureReason, results}, null, 2)}\n`);
}
console.log(`Verified production TimeBar layout at ${results.map(result => result.viewportWidth).join('px and ')}px: ${outputFile}`);
