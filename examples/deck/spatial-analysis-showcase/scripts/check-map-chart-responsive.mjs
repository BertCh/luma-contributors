// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';
import ts from 'typescript';
import {chromium} from 'playwright';

const outputDirectory = process.argv[2] ?? '/tmp/showcase-map-chart-responsive';
const sourceDirectory = fileURLToPath(new URL('../src/', import.meta.url));
const source = await readFile(new URL('../src/shell/story-view.ts', import.meta.url), 'utf8');
const syntax = ts.createSourceFile('story-view.ts', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
const nodes = [];
function visit(node) { nodes.push(node); ts.forEachChild(node, visit); }
visit(syntax);
const method = name => nodes.find(node => ts.isMethodDeclaration(node) && node.name?.getText(syntax) === name)?.getText(syntax);
const readoutSource = await readFile(new URL('../src/shell/story-readouts.ts', import.meta.url), 'utf8');
const readoutSyntax = ts.createSourceFile('story-readouts.ts', readoutSource, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
const readoutHelpers = ['getChartLink', 'formatReadout', 'formatNumber'].map(name => readoutSyntax.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === name)?.getText(readoutSyntax));
assert.ok(readoutHelpers.every(Boolean));
const methodNames = ['placeLegend', 'createReadoutView', 'renderLinkedChart', 'paintReadout', 'getReadoutViews', 'setChart', 'paintInset'];
const methods = methodNames.map(method);
assert.ok(methods.every(Boolean), 'Extract actual production chart publication and readout methods.');
assert.ok(!method('renderStep')?.includes('!isMapChart(spec)') && source.includes('Boolean(spec) && (!spec!.hood || listed.has(spec!.id))'), 'Active-step map charts must have phone readout rows.');
const fixtureSource = `
import {h, formatBytes} from './shell/dom';
import {renderChart} from './shell/chart';
${readoutHelpers.join('\n')}
import {MapChips} from './shell/map-chips';
import {MapInset} from './shell/map-inset';
const isMapChart = spec => spec?.kind === 'chart' && spec.placement === 'map';
export class ChartFixture {
  constructor(corner) {
    this.mapCorners = Object.fromEntries(['topLeft', 'topRight', 'bottomLeft', 'bottomRight'].map(name => [name, document.querySelector('.map-corner-' + name.replace(/[A-Z]/g, letter => '-' + letter.toLowerCase()))]));
    this.legendOverlay = document.querySelector('.map-legend');
    this.chips = new MapChips(); this.chips.setGroundNote('Map: dark ground'); this.mapCorners.topRight.append(this.chips.element);
    this.refresh();
    this.readoutSpecs = new Map(); this.readoutCharts = new Map(); this.readoutValues = new Map();
    this.allReadoutViews = new Map(); this.stepReadoutViews = new Map(); this.insetViews = new Map();
    this.store = {get: () => 0};
    const spec = {id: 'histogram', label: 'Magnitude histogram', kind: 'chart', placement: 'map'};
    this.readoutSpecs.set(spec.id, spec);
    const inset = new MapInset({title: spec.label, corner});
    this.insetViews.set(spec.id, {inset, chartElement: null, linkOption: null});
    document.querySelector('.map-corner-' + corner).append(inset.element);
    const row = this.createReadoutView(spec, false);
    this.stepReadoutViews.set(spec.id, row);
    document.querySelector('.step-readouts').append(row.row);
    this.publish('Initial distribution', [4, 9, 2]);
  }
  publish(title, values) { this.setChart('histogram', {kind: 'histogram', title, values, xDomain: [4, 9], xLabel: 'Magnitude', yLabel: 'Events'}); }
  refresh() { this.placeLegend(window.matchMedia('(max-width: 900px)').matches); }
  scheduleObstacles() {}
  ${methods.join('\n')}
}
`;
const bundle = await build({stdin: {contents: fixtureSource, resolveDir: sourceDirectory, loader: 'ts'}, bundle: true, write: false, format: 'iife', globalName: 'MapChartFixtureModule', platform: 'browser', loader: {'.css': 'empty'}});
const styles = await Promise.all(['styles.css', 'shell/story-view.css', 'shell/legend.css', 'shell/map-furniture.css', 'shell/map-inset.css', 'shell/chart.css', 'shell/map-chips.css'].map(path => readFile(new URL(`../src/${path}`, import.meta.url), 'utf8')));
await mkdir(outputDirectory, {recursive: true});
const reports = [];
const errors = [];
const browser = await chromium.launch({headless: true});
try {
  for (const corner of ['top-right', 'bottom-right']) {
    const page = await browser.newPage({viewport: {width: 1440, height: 960}});
    page.on('pageerror', error => errors.push(error.message));
    await page.setContent(`<body class="is-story"><header class="site-header">Spatial Analysis</header><main class="view"><div class="story-layout"><aside class="story-panel" style="--sheet-h:46dvh"><div class="story-panel-scroll"><h2>Explore the query</h2><p>Observed support and query controls.</p><dl class="readouts step-readouts"></dl></div></aside><div class="story-map"><div class="map-corner map-corner-top-left"><div class="map-furniture-cartouche"><div class="map-furniture-title">Earthquake catalog query lab</div></div><div class="map-legend is-collapsed"><button class="map-legend-toggle">Legend</button><div class="map-legend-body">Magnitude classes and observed support.</div></div></div><div class="map-corner map-corner-top-right"></div><div class="map-corner map-corner-bottom-left"></div><div class="map-corner map-corner-bottom-right"></div><div class="map-credit-slot">US Geological Survey. This is a bounded archive.</div></div></div></main></body>`);
    for (const content of styles) await page.addStyleTag({content});
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(corner => {
      globalThis.fixture = new MapChartFixtureModule.ChartFixture(corner);
      document.querySelector('.map-legend-toggle').addEventListener('click', () => document.querySelector('.map-legend').classList.toggle('is-collapsed'));
    }, corner);
    const inspect = () => page.evaluate(() => {
      const rectangle = selector => {const element = document.querySelector(selector); const box = element.getBoundingClientRect(); return {top: box.top, bottom: box.bottom, left: box.left, right: box.right, width: box.width, height: box.height, visible: Boolean(box.width && box.height)};};
      const toggle = document.querySelector('.map-legend-toggle'); const box = toggle.getBoundingClientRect();
      return {width: innerWidth, documentWidth: document.documentElement.scrollWidth, inset: rectangle('.map-inset'), row: rectangle('.mobile-map-chart-readout'), chart: rectangle('.mobile-map-chart-readout svg'), panel: rectangle('.story-panel'), cartouche: rectangle('.map-furniture-cartouche'), chips: rectangle('.map-chips'), legendHit: toggle.contains(document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2))};
    });
    const desktop = await inspect();
    assert.ok(desktop.inset.visible && !desktop.row.visible);
    for (const width of [390, 320]) {
      await page.setViewportSize({width, height: 844});
      await page.evaluate(() => fixture.refresh());
      const phone = await inspect();
      assert.ok(!phone.inset.visible && phone.row.visible, 'Phone chart stays available in the sheet, with no map overlay.');
      assert.ok(phone.chips.top >= phone.cartouche.bottom, 'Status pills cannot cover the mobile title.');
      assert.ok(phone.legendHit, 'The actual legend hit target cannot be intercepted by an inset SVG.');
      assert.equal(phone.documentWidth, phone.width);
      assert.ok(phone.chart.width > 200 && phone.chart.left >= phone.panel.left && phone.chart.right <= phone.panel.right, 'Chart fits the phone sheet.');
      await page.locator('.map-legend-toggle').click();
      await page.locator('.map-legend-toggle').click();
      await page.evaluate(() => fixture.publish('Updated distribution', [2, 7, 12]));
      assert.ok((await page.locator('.mobile-map-chart-readout').getByText('Updated distribution', {exact: true}).count()) > 0);
      assert.ok((await page.locator('.map-inset').getByText('Updated distribution', {exact: true}).count()) > 0);
      const values = page.locator('.mobile-map-chart-readout').getByText('Show values', {exact: true});
      await values.click();
      assert.ok(await page.locator('.mobile-map-chart-readout table').isVisible(), 'The sheet retains the chart value-table control.');
      await page.screenshot({path: `${outputDirectory}/${corner}-${width}.png`});
      reports.push({corner, desktop, phone});
    }
    await page.setViewportSize({width: 1440, height: 960});
    await page.evaluate(() => fixture.refresh());
    const restored = await inspect();
    assert.ok(restored.inset.visible && !restored.row.visible, 'Desktop corner chart returns after resizing.');
    await page.close();
  }
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({reports, errors}, null, 2) + '\n');
}
console.log(`PASS responsive map charts: actual readout/chart publication, both corners, 390/320 px, legend hit targets, live updates and value tables. ${outputDirectory}`);
