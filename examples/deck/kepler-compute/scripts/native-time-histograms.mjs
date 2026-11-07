// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {runInNewContext} from 'node:vm';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const require = createRequire(import.meta.url);
const nativeRequire = createRequire(require.resolve('@kepler.gl/utils'));
const directory = join(dirname(require.resolve('@kepler.gl/utils')), '../esm');
function extract(file, name) {
  const source = readFileSync(join(directory, file), 'utf8');
  const start = source.indexOf(`function ${name}(`);
  assert(start >= 0, `Missing pinned native function ${name}`);
  return source.slice(start, source.indexOf('\n}', start) + 2);
}
const native = runInNewContext([
  extract('time.js', 'intervalToFunction'), extract('plot.js', 'getBinThresholds'),
  extract('plot.js', 'histogramFromThreshold'), '({getBinThresholds,histogramFromThreshold})'
].join('\n'), {
  TIME_INTERVALS: nativeRequire('@kepler.gl/constants').TIME_INTERVALS,
  durationDay: 86400000, moment: nativeRequire('moment'),
  d3Histogram: nativeRequire('d3-array').histogram, _slicedToArray: value => value
});
const instants = [
  '1969-12-31T00:00:00Z', '1970-01-01T00:00:00Z', '1970-02-01T00:00:00Z',
  '1970-02-28T00:00:00Z', '1970-03-01T00:00:00Z', '1970-03-15T00:00:00Z',
  '1970-04-01T00:00:00Z'
].map(Date.parse).concat(null);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless:true, backend:'webgpu-core'}));
const page = await browser.newPage({viewport:{width:1600,height:1100}});
page.setDefaultTimeout(30000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => {errors.push(error.message);console.error(error.stack);});
page.on('console', message => {if(message.type()==='error')console.error(message.text());});
let datasetId;
let filterId;
const settings = page.locator('.time-widget--settings');
async function assertBins(interval, rows) {
  await page.waitForFunction(({filterId,interval}) => {
    const filter=window.__keplerComputeTest.getState().nativeFilters.find(filter=>filter.id===filterId);
    return filter?.plotType?.interval===interval && filter.lumaComputePlot?.status==='ready';
  }, {filterId,interval});
  const filter=await page.evaluate(filterId=>window.__keplerComputeTest.getState().nativeFilters.find(filter=>filter.id===filterId),filterId);
  const expected=native.histogramFromThreshold(native.getBinThresholds(interval,filter.domain),rows,index=>instants[index])
    .map(({x0,x1,count})=>({x0,x1,count}));
  const actual=filter.timeBins[datasetId][interval].map(({x0,x1,count,indexes})=>{
    assert.deepEqual(indexes,[], 'Resident native histograms must not expose source-row arrays');
    return {x0,x1,count};
  });
  assert.deepEqual(actual,expected);
  assert.equal(actual.reduce((sum,bin)=>sum+bin.count,0),rows.length);
  assert.equal(await page.evaluate(()=>window.__keplerComputeTest.getState().publications[0].selectedCount),3);
  assert.equal(await page.evaluate(()=>window.__keplerComputeTest.getState().error),'');
  return actual;
}
async function chooseInterval(label, interval, rows) {
  await settings.locator('.item-selector__dropdown').first().click();
  await page.getByText(label,{exact:true}).last().click();
  return assertBins(interval,rows);
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.__keplerComputeTest?.getState().selectedCount===256,undefined,{timeout:60000});
  const source=new arrow.Table({time:arrow.vectorFromArray(instants,new arrow.TimestampMillisecond('UTC')),
    rank:arrow.makeVector(new Float32Array([0,1,2,3,4,5,6,7]))});
  await openNativeData(page);
  const workspace=page.getByRole('region',{name:'Resident datasets',exact:true});
  await workspace.locator('input[type=file]').setInputFiles({name:'native-time-histogram.arrow',mimeType:'application/vnd.apache.arrow.stream',buffer:Buffer.from(arrow.tableToIPC(source.slice(0,3).concat(source.slice(3)),'stream'))});
  await workspace.getByRole('button',{name:'Check schema and precision',exact:true}).click();
  await workspace.getByRole('button',{name:'Publish selected fields',exact:true}).click();
  await page.waitForFunction(()=>window.__keplerComputeTest.getState().publications[0]?.selectedCount===8);
  const initial=await page.evaluate(()=>window.__keplerComputeTest.getState().publications[0]);datasetId=initial.id;
  await closeNativeDialog(page);
  await page.getByRole('button',{name:'Toggle charts',exact:true}).click();
  await page.locator('.chart-config__type').click();
  await page.getByRole('dialog').getByText('Line',{exact:true}).click();
  const card=page.locator('.resident-chart-card').first();
  await card.getByLabel('Chart dataset',{exact:true}).selectOption(datasetId);
  await card.getByLabel('Chart categories',{exact:true}).selectOption('time');
  await card.getByLabel('Chart binning',{exact:true}).selectOption('timeBin');
  await card.getByLabel('Calendar interval (UTC)',{exact:true}).selectOption('month');
  await card.getByLabel('Chart measure',{exact:true}).selectOption('count');
  await card.getByRole('button',{name:'1970-02: 2',exact:true}).click();
  await page.waitForFunction(()=>window.__keplerComputeTest.getState().publications[0]?.selectedCount===3);
  filterId=await page.evaluate(datasetId=>window.__keplerComputeTest.getState().nativeFilters.find(filter=>filter.type==='timeRange'&&filter.dataId.includes(datasetId)).id,datasetId);
  await card.getByLabel('Keep chart visible', {exact:true}).uncheck();
  // The chart pane overlaps the map control; use its native keyboard activation to close it.
  await page.getByRole('button',{name:'Toggle charts',exact:true}).press('Enter');
  await card.waitFor({state:'hidden'});
  await page.locator('.side-panel__tab[data-for="filter-nav"]').click();
  await page.locator(`[data-for=${JSON.stringify(`tooltip.timePlayback_${filterId}`)}]`).click();
  await page.waitForFunction(filterId=>window.__keplerComputeTest.getState().nativeFilters.find(filter=>filter.id===filterId)?.view==='enlarged',filterId);
  await page.locator('.time-widget--top .data-ex-icons-gear').last().click();
  await settings.waitFor();
  const allRows=instants.map((_,index)=>index);
  const monthBins=await chooseInterval('Month','1-month',allRows);
  assert.deepEqual(monthBins.map(bin=>bin.count),[1,2,2,2,1]);
  await chooseInterval('Year','1-year',allRows);
  const selectedRows=[2,3,4,5,6];
  await page.evaluate(datasetId=>window.__keplerComputeTest.setPublishedRange(datasetId,'rank',2,6),datasetId);
  await assertBins('1-year',selectedRows);
  await chooseInterval('Month','1-month',selectedRows);
  // Actual native histogram SVG consumes the supplied calendar bins; this is not a custom chart.
  const widget=page.locator('.bottom-widget--inner').last();
  const visibleBars=await widget.locator('.histogram-bars rect').evaluateAll(rectangles=>rectangles.filter(rectangle=>Number(rectangle.getAttribute('height'))>0&&Number(rectangle.getAttribute('width'))>0).length);
  assert(visibleBars>=3,'Native time histogram must draw nonempty month bars');
  await page.screenshot({path:'/tmp/kepler-native-time-histograms.png'});
  const saved=await page.evaluate(()=>window.__keplerComputeTest.saveNativeConfig());
  assert.equal(saved.config.visState.filters.find(filter=>filter.id===filterId).plotType.interval,'1-month');
  await page.evaluate(config=>window.__keplerComputeTest.applyNativeConfig(config),saved);
  await assertBins('1-month',selectedRows);
  const final=await page.evaluate(()=>window.__keplerComputeTest.getState().publications[0]);
  for(const name of ['sourceRevision','sourceUploadBytes','sourceBufferIds','sourceBatches'])assert.deepEqual(final[name],initial[name]);
  assert.deepEqual(initial.sourceBatches,[3,5]);assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,checks:['native chart-created time filter','actual time-widget Month/Year interval dropdown','pinned native edge/count oracle','null-at-epoch histogram parity','self-filter exclusion with narrowed map count','unrelated rank filter retained','native histogram SVG bars','saved interval replay','unchanged source revision/uploads/buffers/chunks']}));
} catch(error) {
  console.error(await page.evaluate(()=>({state:window.__keplerComputeTest?.getState(),text:document.body.innerText})).catch(()=> 'Page unavailable'));
  throw error;
} finally {await browser.close();}
