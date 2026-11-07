// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {build} from 'esbuild';
import ts from 'typescript';
import {chromium} from 'playwright';

const outputDirectory = process.argv[2] ?? '/tmp/showcase-responsive-camera';
const source = await readFile(new URL('../src/shell/story-view.ts', import.meta.url), 'utf8');
const syntax = ts.createSourceFile('story-view.ts', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
const nodes = [];
function visit(node) { nodes.push(node); ts.forEachChild(node, visit); }
visit(syntax);
const method = name => nodes.find(node => ts.isMethodDeclaration(node) && node.name?.getText(syntax) === name)?.getText(syntax);
const helper = name => nodes.find(node => ts.isFunctionDeclaration(node) && node.name?.getText(syntax) === name)?.getText(syntax);
const variable = name => nodes.find(node => ts.isVariableDeclaration(node) && node.name.getText(syntax) === name)?.parent.parent.getText(syntax);
const methodNames = ['setupResponsiveCamera', 'armResponsiveBoundsRefit', 'scheduleResponsiveBoundsRefit', 'cancelResponsiveBoundsRefit', 'interruptResponsiveBoundsRefit', 'getEffectiveCamera', 'moveToStepCamera', 'updateViewPadding', 'checkCameraMoved', 'setCameraMoved'];
assert.ok(methodNames.every(name => method(name)), 'Extract actual production methods.');
assert.ok(method('unmount').includes('cancelResponsiveBoundsRefit'), 'Unmount cancels pending callbacks.');
assert.ok(method('moveToStepCamera').includes('cancelResponsiveBoundsRefit'), 'Explicit step navigation cancels pending callbacks.');
const fields = nodes.filter(node => ts.isPropertyDeclaration(node) && node.name.getText(syntax).startsWith('responsiveBoundsRefit')).map(node => node.getText(syntax));
const fixtureSource = `
const prefersReducedMotion = () => true;
${['DEFAULT_FLY_MILLISECONDS', 'SHEET_CLEARANCE_PIXELS', 'MOVED_PIXELS', 'MOVED_ZOOM', 'MOVED_DEGREES'].map(variable).join('\n')}
${['normalizePadding', 'addPadding', 'getMercatorY'].map(helper).join('\n')}
export class CameraFixture {
  ${fields.join('\n')}
  constructor(properties) {
    Object.assign(this, properties);
    this.mountGeneration = 1;
    this.stepIndex = 0;
    this.cameraMoved = false;
    this.cameraSettleTimer = 0;
    this.paddingTimer = 0;
    this.stepViewReference = this.dependencies.getHost().getViewState();
  }
  scheduleObstacles() {}
  updateLegendLayout() {}
  installResize() {
    const sheet = {refresh: () => this.updateViewPadding(true)};
    ${variable('onResize')}
    window.addEventListener('resize', onResize);
    const cleanupInput = this.setupResponsiveCamera(this.dependencies.getHost());
    return () => {window.removeEventListener('resize', onResize); cleanupInput();};
  }
  updateModifiedChip() {}
  ${methodNames.map(method).join('\n')}
}
`;
const bundle = await build({stdin: {contents: fixtureSource, loader: 'ts'}, bundle: true, write: false, format: 'iife', globalName: 'CameraFixtureModule', platform: 'browser'});
await mkdir(outputDirectory, {recursive: true});
const browser = await chromium.launch({headless: true});
const reports = [];
const errors = [];
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 960}, reducedMotion: 'reduce'});
  page.on('pageerror', error => errors.push(error.message));
  await page.setContent('<div id="map" style="height:800px;width:100%" tabindex="0"></div>');
  await page.addScriptTag({content: bundle.outputFiles[0].text});
  async function setup(camera, moved = false) {
    await page.evaluate(({camera, moved}) => {
      globalThis.cleanup?.();
      if (globalThis.fixture) {
        clearTimeout(fixture.cameraSettleTimer);
        clearTimeout(fixture.paddingTimer);
      }
      const element = document.querySelector('#map');
      const calls = [];
      let view = {longitude: 0, latitude: 20, zoom: 2, pitch: 0, bearing: 0};
      let padding = {top: 0, bottom: 0, left: 0, right: 0};
      const host = {
        element,
        fitBounds: (bounds, options) => {
          calls.push({kind: 'fit', bounds, options, padding, width: innerWidth});
          view = {...view, zoom: innerWidth < 900 ? 0 : 2};
        },
        getViewState: () => ({...view}),
        setViewState: value => {calls.push({kind: 'view', value}); view = {...view, ...value};},
        getViewPadding: () => ({...padding}),
        setViewPadding: value => {padding = {...padding, ...value}; calls.push({kind: 'padding', padding});}
      };
      globalThis.calls = calls;
      globalThis.fixture = new CameraFixtureModule.CameraFixture({
        scene: {id: 'fixture', initialView: view, story: [{camera}, {}]},
        dependencies: {hasWebGPU: true, getHost: () => host},
        mapSlot: element,
        sheet: {getHeight: () => innerWidth < 900 ? 300 : 0}
      });
      fixture.cameraMoved = moved;
      globalThis.cleanup = fixture.installResize();
    }, {camera, moved});
  }
  const bounds = {bounds: [-170, -55, 170, 75], zoom: 5, pitch: 0, bearing: 0, padding: 24};
  const settle = () => page.waitForTimeout(550);
  const fitCalls = () => page.evaluate(() => calls.filter(call => call.kind === 'fit'));
  await setup(bounds);
  await page.setViewportSize({width: 390, height: 844});
  await settle();
  const phoneFits = await fitCalls();
  assert.equal(phoneFits.length, 1);
  assert.equal(phoneFits[0].width, 390);
  assert.equal(phoneFits[0].padding.bottom, 316);
  assert.deepEqual(phoneFits[0].bounds, bounds.bounds);
  assert.deepEqual(phoneFits[0].options, {padding: {top: 24, right: 24, bottom: 24, left: 24}, maxZoom: 5, transitionMs: 0});
  reports.push({case: 'desktop-to-phone bounds after standing padding settles', calls: phoneFits});

  await setup(bounds);
  for (const width of [420, 430, 440]) await page.setViewportSize({width, height: 844});
  await settle();
  assert.equal((await fitCalls()).length, 1, 'Rapid same-breakpoint resize coalesces into one fit.');
  await settle();
  assert.equal((await fitCalls()).length, 1, 'No fitting loop after settling.');
  reports.push({case: 'same-breakpoint resize burst coalesces without frame loops'});

  await setup({longitude: 30, latitude: 20, zoom: 4});
  await page.setViewportSize({width: 450, height: 844}); await settle();
  assert.equal((await fitCalls()).length, 0);
  assert.equal(await page.evaluate(() => calls.filter(call => call.kind === 'view').length), 0);
  reports.push({case: 'explicit view is not fitted or rewritten'});

  await setup(bounds, true);
  await page.setViewportSize({width: 460, height: 844}); await settle();
  assert.equal((await fitCalls()).length, 0);
  reports.push({case: 'already explored camera is preserved'});

  for (const eventType of ['pointerdown', 'wheel', 'keydown']) {
    await setup(bounds);
    await page.evaluate(() => {dispatchEvent(new Event('resize')); fixture.stepViewReference = null;});
    await page.evaluate(eventType => {
      const event = eventType === 'keydown' ? new KeyboardEvent(eventType, {key: 'ArrowRight', bubbles: true}) : new Event(eventType, {bubbles: true});
      fixture.dependencies.getHost().element.dispatchEvent(event);
      // A later event in the same resize burst must not re-arm the cancelled fit.
      dispatchEvent(new Event('resize'));
    }, eventType);
    await settle();
    assert.equal((await fitCalls()).length, 0);
    reports.push({case: `${eventType} cancels pending fit during padding transition`});
  }

  await page.evaluate(() => {fixture.moveToStepCamera(false); dispatchEvent(new Event('resize'));});
  await settle();
  assert.equal((await fitCalls()).length, 2, 'Explicit camera reset restores responsive fitting after an interrupted resize.');
  reports.push({case: 'explicit camera reset restores responsive fitting'});

  for (const change of ['camera', 'step', 'mount', 'cleanup']) {
    await setup(bounds);
    await page.evaluate(change => {
      dispatchEvent(new Event('resize'));
      if (change === 'camera') fixture.cameraMoved = true;
      if (change === 'step') fixture.stepIndex = 1;
      if (change === 'mount') fixture.mountGeneration++;
      if (change === 'cleanup') {cleanup(); dispatchEvent(new Event('resize'));}
    }, change);
    await settle();
    assert.equal((await fitCalls()).length, 0);
    reports.push({case: `${change} invalidates pending fit`});
  }
  await page.evaluate(() => cleanup());
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({reports, errors}, null, 2));
}
console.log(`PASS ${reports.length} responsive bounds cases using actual StoryView methods and a mocked host; no GPU. ${outputDirectory}`);
