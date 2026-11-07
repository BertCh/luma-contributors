// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';
import {chromium} from 'playwright';

const outputDirectory = process.argv[2] ?? '/tmp/showcase-map-legend-spacing';
const sourceDirectory = fileURLToPath(new URL('../src/', import.meta.url));
const bundle = await build({
  stdin: {contents: "export {MapFurniture, createScaleRow} from './shell/map-furniture'; export {renderLegends} from './shell/legend';", resolveDir: sourceDirectory, loader: 'ts'},
  bundle: true, write: false, format: 'iife', globalName: 'LegendFixtureModule', platform: 'browser', loader: {'.css': 'empty'},
  plugins: [{name: 'expose-actual-scale-row', setup(buildContext) {
    buildContext.onLoad({filter: /\/shell\/map-furniture\.ts$/}, async args => ({contents: `${await readFile(args.path, 'utf8')}\nexport {createScaleRow};`, loader: 'ts'}));
  }}]
});
const styles = await Promise.all(['styles.css', 'shell/legend.css', 'shell/map-furniture.css'].map(path => readFile(new URL(`../src/${path}`, import.meta.url), 'utf8')));
await mkdir(outputDirectory, {recursive: true});
const browser = await chromium.launch({headless: true});
const errors = [];
let report;
try {
  const page = await browser.newPage({viewport: {width: 700, height: 1100}});
  page.on('pageerror', error => errors.push(error.message));
  await page.setContent('<main style="padding:24px;display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start"></main>');
  for (const content of styles) await page.addStyleTag({content});
  await page.addScriptTag({content: bundle.outputFiles[0].text});
  report = await page.evaluate(() => {
    const root = document.querySelector('main');
    const cases = [];
    for (const fontSize of [10.5, 15]) {
      for (const width of [40, 60, 80, 120]) {
        for (const numerals of [['0', '500', '1,000 km'], ['0', '250', '500 m'], ['0', '10', '20 km']]) {
          const container = document.createElement('div');
          container.className = 'map-furniture-scale';
          container.style.fontSize = `${fontSize}px`;
          root.append(container);
          const row = LegendFixtureModule.createScaleRow({label: numerals[2], widthPixels: width, numerals, ticks: []}, container);
          container.append(row);
          const labels = [...row.querySelectorAll('.map-furniture-scale-numerals span')].map(element => {
            const rectangle = element.getBoundingClientRect();
            return {text: element.textContent, left: rectangle.left, right: rectangle.right, top: rectangle.top, bottom: rectangle.bottom};
          });
          cases.push({fontSize, width, font: getComputedStyle(container).font, family: getComputedStyle(container).fontFamily, weight: getComputedStyle(container).fontWeight, labels, stacked: Boolean(row.querySelector('.is-stacked'))});
        }
      }
    }
    const furniture = new LegendFixtureModule.MapFurniture();
    furniture.setSpec({scaleBar: true}, {sceneTitle: 'Fixture', stepTitle: 'Fixture', credits: []});
    const view = {latitude: 40, zoom: 13, bearing: 0, pitch: 0};
    furniture.updateView(view, 40);
    root.append(furniture.scaleBar);
    for (const fontSize of [15, 18]) {
      furniture.scaleBar.style.fontSize = `${fontSize}px`;
      furniture.updateView(view, 40);
      const row = furniture.scaleBar.querySelector('.map-furniture-scale-row');
      const labels = [...row.querySelectorAll('.map-furniture-scale-numerals span')].map(element => {
        const rectangle = element.getBoundingClientRect();
        return {text: element.textContent, left: rectangle.left, right: rectangle.right, top: rectangle.top, bottom: rectangle.bottom};
      });
      cases.push({integration: 'detached-to-connected and typography change with same camera', fontSize, width: row.getBoundingClientRect().width, labels, stacked: Boolean(row.querySelector('.is-stacked'))});
    }
    const legend = document.createElement('div');
    legend.className = 'map-legend';
    legend.style.position = 'relative';
    root.append(legend);
    LegendFixtureModule.renderLegends(legend, [{kind: 'size', title: '1990 population', unit: 'people', layout: 'nested', color: [48, 130, 167, 255], outline: [30, 65, 80, 255], entries: [{label: '1000k', radiusPixels: 46}, {label: '500k', radiusPixels: 32.5}, {label: '250k', radiusPixels: 23}]}], new Map());
    const svg = legend.querySelector('svg');
    const textBounds = [...svg.querySelectorAll('text')].map(element => {const rectangle = element.getBBox(); return {text: element.textContent, x: rectangle.x, y: rectangle.y, width: rectangle.width, height: rectangle.height};});
    const circles = [...svg.querySelectorAll('circle')].map(element => ({radius: Number(element.getAttribute('r')), centerY: Number(element.getAttribute('cy'))}));
    const viewBox = svg.viewBox.baseVal;
    const tickCases = [];
    for (const width of [60, 12]) {
      const container = document.createElement('div');
      container.className = 'map-furniture-scale';
      container.style.fontSize = '15px';
      root.append(container);
      const row = LegendFixtureModule.createScaleRow({
        label: '500 m', widthPixels: width, numerals: ['0', '250', '500 m'],
        ticks: [{meters: 500, label: '500 m', offsetPixels: width}]
      }, container);
      container.append(row);
      const bar = row.querySelector('.map-furniture-scale-bar').getBoundingClientRect();
      const tick = row.querySelector('.map-furniture-scale-tick').getBoundingClientRect();
      const label = row.querySelector('.map-furniture-scale-tick-label')?.getBoundingClientRect();
      tickCases.push({width, bar: {left: bar.left, right: bar.right}, tickCenter: tick.left + tick.width / 2,
        label: label ? {left: label.left, right: label.right} : null,
        endpointNumeral: row.querySelector('.map-furniture-scale-numerals > .is-end')?.textContent});
    }
    return {cases, tickCases, nested: {textBounds, circles, viewBox: {x: viewBox.x, y: viewBox.y, width: viewBox.width, height: viewBox.height}}};
  });
  for (const item of report.cases) {
    assert.equal(item.labels[0].text, '0');
    assert.ok(item.labels.at(-1).text.includes('m'));
    for (let first = 0; first < item.labels.length; first++) {
      for (let second = first + 1; second < item.labels.length; second++) {
        const a = item.labels[first]; const b = item.labels[second];
        assert.ok(a.right <= b.left || b.right <= a.left || a.bottom <= b.top || b.bottom <= a.top, `Text collision ${JSON.stringify(item)}`);
      }
    }
  }
  assert.ok(report.cases.some(item => item.labels.length === 3), 'Readable wider scales retain a midpoint.');
  assert.ok(report.cases.some(item => item.stacked), 'Tiny bars retain zero and endpoint on separate lines.');
  const [rightEdgeTick, narrowTick] = report.tickCases;
  assert.ok(Math.abs(rightEdgeTick.tickCenter - rightEdgeTick.bar.right) <= 1, 'The tick mark stays at its true right-edge offset.');
  assert.ok(rightEdgeTick.label, 'A fitting right-edge tick retains its label.');
  assert.ok(rightEdgeTick.label.left >= rightEdgeTick.bar.left && rightEdgeTick.label.right <= rightEdgeTick.bar.right, 'The right-edge label stays inside the bar.');
  assert.equal(narrowTick.label, null, 'A label wider than a narrow bar is omitted while its tick mark remains.');
  assert.ok(Math.abs(narrowTick.tickCenter - narrowTick.bar.right) <= 1, 'The narrow-bar endpoint mark stays at its true offset.');
  assert.equal(narrowTick.endpointNumeral, '500 m', 'The scale endpoint still labels the distance when its accent label is omitted.');
  for (const text of report.nested.textBounds) {
    const bounds = report.nested.viewBox;
    assert.ok(text.x >= bounds.x && text.y >= bounds.y && text.x + text.width <= bounds.x + bounds.width && text.y + text.height <= bounds.y + bounds.height, `SVG clips ${JSON.stringify(text)}`);
  }
  assert.deepEqual(report.nested.circles.map(circle => circle.radius), [46, 32.5, 23], 'Typography fix must not change proportional areas.');
  await page.screenshot({path: `${outputDirectory}/legend-spacing.png`, fullPage: true});
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({report, errors}, null, 2));
}
console.log(`PASS ${report.cases.length} actual DOM scale layouts, ${report.tickCases.length} endpoint tick cases and all nested SVG text bounds; no GPU. ${outputDirectory}`);
