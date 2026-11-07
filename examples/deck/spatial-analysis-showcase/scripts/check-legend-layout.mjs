// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';
import ts from 'typescript';
import {chromium} from 'playwright';

const outputDirectory = process.argv[2] ?? '/tmp/showcase-legend-layout';
const sourceDirectory = fileURLToPath(new URL('../src/', import.meta.url));
const source = await readFile(new URL('../src/shell/story-view.ts', import.meta.url), 'utf8');
const syntax = ts.createSourceFile('story-view.ts', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
const nodes = [];
function visit(node) { nodes.push(node); ts.forEachChild(node, visit); }
visit(syntax);
const method = name => nodes.find(node => ts.isMethodDeclaration(node) && node.name?.getText(syntax) === name)?.getText(syntax);
const variable = name => nodes.find(node => ts.isVariableDeclaration(node) && node.name.getText(syntax) === name)?.parent.parent.getText(syntax);
const statement = prefix => nodes.find(node => ts.isExpressionStatement(node) && node.getText(syntax).startsWith(prefix))?.getText(syntax);
const pieces = [method('updateLegendLayout'), method('placeLegend'), variable('userLegendPreference'), variable('setLegendCollapsed'), statement('this.legendToggle.addEventListener'), statement('this.panelLegendButton.addEventListener'), statement('openPanelLegend =')];
assert.ok(pieces.every(Boolean), 'Fixture extracts actual production placement and interaction code, never a duplicate implementation.');
const fixtureSource = `
import {getLegendLayoutMode} from './shell/legend-layout';
import {SheetController} from './shell/story-sheet';
const MOBILE_QUERY = '(max-width: 900px)';
export class LegendFixture {
  constructor(properties) {
    Object.assign(this, properties);
    let openPanelLegend = () => {};
    ${variable('userLegendPreference')}
    ${variable('setLegendCollapsed')}
    setLegendCollapsed(window.matchMedia(MOBILE_QUERY).matches);
    ${statement('this.legendToggle.addEventListener')}
    ${statement('this.panelLegendButton.addEventListener')}
    ${statement('openPanelLegend =')}
    this.getPreference = () => userLegendPreference;
    this.sheet = new SheetController({panel: this.panel, handle: this.handle, mapSlot: this.mapSlot,
      onSettle: () => this.updateLegendLayout()});
    this.refresh();
    this.observerCount = 0;
    this.observer = new ResizeObserver(() => {this.observerCount++; this.updateLegendLayout();});
    this.observer.observe(this.mapSlot); this.observer.observe(this.mapCorners.topLeft); this.observer.observe(this.mapCorners.bottomLeft);
  }
  scheduleObstacles() {}
  refresh() {
    const mobile = window.matchMedia(MOBILE_QUERY).matches;
    this.placeLegend(mobile); this.sheet.setEnabled(mobile); this.sheet.refresh(); this.updateLegendLayout();
  }
  ${method('placeLegend')}
  ${method('updateLegendLayout')}
}
`;
const bundle = await build({stdin: {contents: fixtureSource, resolveDir: sourceDirectory, loader: 'ts'}, bundle: true, write: false, format: 'iife', globalName: 'LegendFixtureModule', platform: 'browser'});
const styles = await Promise.all(['styles.css', 'shell/story-view.css', 'shell/legend.css', 'shell/map-furniture.css', 'shell/map-inset.css'].map(path => readFile(new URL(`../src/${path}`, import.meta.url), 'utf8')));
await mkdir(outputDirectory, {recursive: true});
const browser = await chromium.launch({headless: true});
const reports = [];
const errors = [];
try {
  for (const scenario of [{name: 'desktop', width: 1440, height: 960, timeline: false}, {name: 'phone', width: 390, height: 844, timeline: false}, {name: 'compact-phone-timeline', width: 320, height: 640, timeline: true}, {name: 'short-phone-timeline', width: 320, height: 568, timeline: true}]) {
    const page = await browser.newPage({viewport: {width: scenario.width, height: scenario.height}, reducedMotion: 'reduce'});
    page.on('pageerror', error => errors.push(error.message));
    await page.setContent(`<body class="is-story"><header class="site-header"><a class="logo">Spatial Analysis</a></header><main class="view"><div class="story-layout"><aside class="story-panel"><button class="sheet-handle" aria-label="Expand or collapse the story panel"><span></span></button><div class="sheet-peek"><button class="sheet-peek-button">Back</button><span class="sheet-peek-title">Measured observations</span><button class="sheet-peek-button">Next</button></div><div class="story-panel-scroll"><h2>Measured observations</h2><p>Reading panel and complete source information.</p></div><footer class="panel-footer"><div class="step-progress"></div><div class="panel-footer-row"><button class="btn">Back</button><div class="panel-footer-tools"><button class="btn btn-ghost btn-small">Auto-play tour</button><button class="btn btn-ghost btn-small panel-legend-button" hidden aria-controls="map-legend-overlay">Legend</button></div><button class="btn btn-primary">Next</button></div></footer></aside><div id="story-map" class="story-map ${scenario.timeline ? 'has-time-bar' : ''}" style="--map-credit-height:40px"><div class="map-corner map-corner-top-left"><div class="map-furniture-cartouche"><div class="map-furniture-title">A measured time series</div><div>Observed acquisition support</div></div></div><div class="map-corner map-corner-bottom-left"></div><div class="map-credit-slot" style="height:40px">Source attribution and a meaningful caveat</div><div id="map-legend-overlay" class="map-legend"><button class="map-legend-toggle" aria-expanded="true">Legend</button><div class="map-legend-body"><div class="legends">${'<p>Source and a meaningful legend class</p>'.repeat(25)}</div></div></div></div></div></main></body>`);
    for (const content of styles) await page.addStyleTag({content});
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => {
      const find = selector => document.querySelector(selector);
      globalThis.fixture = new LegendFixtureModule.LegendFixture({mapSlot: find('#story-map'), creditSlot: find('.map-credit-slot'), panel: find('.story-panel'), handle: find('.sheet-handle'), mapCorners: {topLeft: find('.map-corner-top-left'), bottomLeft: find('.map-corner-bottom-left')}, legendOverlay: find('.map-legend'), legendToggle: find('.map-legend-toggle'), legendContainer: find('.legends'), panelLegendButton: find('.panel-legend-button')});
    });
    const inspect = () => page.evaluate(() => {
      const box = selector => {const element = document.querySelector(selector); const bounds = element.getBoundingClientRect(); return {top: bounds.top, bottom: bounds.bottom, width: bounds.width, height: bounds.height, scrollWidth: element.scrollWidth, visible: Boolean(bounds.width && bounds.height)};};
      return {mode: fixture.legendLayoutMode, detent: fixture.sheet.getDetent(), preference: fixture.getPreference(), collapsed: fixture.legendOverlay.classList.contains('is-collapsed'), observerCount: fixture.observerCount, toggle: box('.map-legend-toggle'), legend: box('.map-legend'), body: box('.map-legend-body'), footer: box('.panel-footer'), panelButton: box('.panel-legend-button'), credit: box('.map-credit-slot'), header: box('.site-header'), map: box('.story-map'), cartouche: box('.map-furniture-cartouche'), viewportHeight: innerHeight, documentWidth: document.documentElement.scrollWidth, viewportWidth: innerWidth};
    });
    const assertHeaderClearance = state => {
      assert.ok(Math.abs(state.header.top) < 1, 'The production header stays at the viewport top.');
      assert.ok(Math.abs(state.map.top - state.header.bottom) < 1, 'The map starts below the production header.');
      assert.ok(Math.abs(state.map.height - (state.viewportHeight - state.header.bottom)) < 1, 'Map height excludes the header.');
      assert.ok(Math.abs(state.map.bottom - state.viewportHeight) < 1, 'The map ends at the viewport bottom.');
      if (state.cartouche.visible) {
        assert.ok(state.cartouche.top >= state.map.top, 'Visible cartouche must not sit behind the header.');
        assert.ok(state.cartouche.bottom <= state.map.bottom, 'Visible cartouche stays inside the map.');
      }
    };
    await page.waitForTimeout(100);
    const initial = await inspect();
    assertHeaderClearance(initial);
    assert.equal(initial.documentWidth, initial.viewportWidth);
    assert.ok(initial.observerCount < 20, 'Overlay layout must settle, not oscillate in ResizeObserver.');
    if (scenario.name === 'short-phone-timeline') {
      assert.equal(initial.mode, 'panel');
      assert.ok(initial.panelButton.visible && initial.panelButton.height >= 44);
      assert.equal(initial.legend.visible, false);
      await page.locator('.panel-legend-button').click();
      await page.waitForTimeout(300);
      const opened = await inspect();
      assert.equal(opened.detent, 'peek');
      assert.equal(opened.preference, false);
      assert.equal(opened.collapsed, false);
      assert.ok(opened.legend.visible && opened.toggle.height >= 44);
      assert.ok(opened.body.height >= 44, 'Opening the legend must provide meaningful scrollable body space.');
      assert.ok(opened.body.bottom <= opened.credit.top);
      await page.screenshot({path: `${outputDirectory}/${scenario.name}-open.png`});
      // The real sheet's handle goes from peek to half to full. Layout must preserve the explicit open preference.
      await page.locator('.sheet-handle').click();
      await page.locator('.sheet-handle').click();
      await page.waitForTimeout(50);
      const full = await inspect();
      assert.equal(full.detent, 'full'); assert.equal(full.mode, 'panel'); assert.equal(full.preference, false);
      assert.ok(full.panelButton.visible);
      await page.locator('.panel-legend-button').click(); await page.waitForTimeout(300);
      await page.locator('.map-legend-toggle').click();
      assert.equal((await inspect()).preference, true);
      await page.setViewportSize({width: 1440, height: 960}); await page.evaluate(() => fixture.refresh());
      const desktopAfterPreference = await inspect();
      assertHeaderClearance(desktopAfterPreference);
      assert.equal(desktopAfterPreference.mode, 'normal'); assert.equal(desktopAfterPreference.collapsed, true);
      reports.push({scenario, initial, opened, full, desktopAfterPreference});
    } else if (scenario.name === 'compact-phone-timeline') {
      assert.equal(initial.mode, 'compact');
      assert.ok(initial.toggle.height >= 44);
      assert.ok(initial.toggle.bottom <= initial.credit.top);
      await page.locator('.map-legend-toggle').click();
      await page.waitForTimeout(300);
      const opened = await inspect();
      assert.equal(opened.detent, 'peek');
      assert.equal(opened.collapsed, false);
      assert.ok(opened.body.height >= 44);
      assert.ok(opened.body.bottom <= opened.credit.top);
      reports.push({scenario, initial, opened});
    } else {
      assert.equal(initial.mode, 'normal');
      assert.equal(initial.panelButton.visible, false);
      assert.ok(initial.toggle.height >= 44);
      assert.ok(initial.legend.bottom <= initial.credit.top);
      if (scenario.name === 'phone') {
        await page.locator('.map-legend-toggle').click();
        const opened = await inspect();
        assert.ok(opened.body.height >= 44 && opened.body.bottom <= opened.credit.top);
        assert.equal(opened.detent, 'half');
      }
      reports.push({scenario, initial});
    }
    await page.screenshot({path: `${outputDirectory}/${scenario.name}.png`});
    // Exercise the actual containing block across the breakpoint, with no fixture-only positioning.
    await page.setViewportSize(scenario.width > 900 ? {width: 390, height: 844} : {width: 1440, height: 960});
    await page.evaluate(() => fixture.refresh());
    await page.waitForTimeout(250);
    const afterResize = await inspect();
    assertHeaderClearance(afterResize);
    reports.at(-1).afterResize = afterResize;
    await page.close();
  }
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({reports, errors}, null, 2));
}
console.log(`PASS actual legend layout/toggle handlers and SheetController: desktop, phone, short timeline phone, footer access, peek/full interaction and explicit preference. ${outputDirectory}`);
