// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {PNG} from 'pngjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const positions = [[-74,40.74],[-73.96,40.76],[-73.98,40.75]];
const batch = indexes => new arrow.Table({
  longitude: arrow.makeVector(Float32Array.from(indexes, index => positions[index][0])),
  latitude: arrow.makeVector(Float32Array.from(indexes, index => positions[index][1])),
  value: arrow.makeVector(Float32Array.from(indexes, index => index + 1)),
  name: arrow.vectorFromArray(indexes.map(index => ['ALPHA','BETA',null][index]), new arrow.Utf8())
});
const source = batch([0,1]).concat(batch([]),batch([2]));
const bytes = arrow.tableToIPC(source,'stream');
assert.deepEqual(arrow.tableFromIPC(bytes).batches.map(value => value.numRows),[2,0,1]);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless:true,backend:'webgpu-core'}));
const page = await browser.newPage({viewport:{width:1440,height:1000}});
page.setDefaultTimeout(30000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.stack ?? error.message));
page.on('console', message => {
  if(message.type()==='error' && /WGSL|validation error|Invalid.*(?:Buffer|Pipeline)/i.test(message.text())) errors.push(message.text());
});
const watchdog = setTimeout(async()=>{
  console.error('Point label browser exceeded three minutes');
  await Promise.race([browser.close(),new Promise(resolve=>setTimeout(resolve,3000))]);
  process.exit(1);
},180000);
let datasetId;
const layerId='native-point-labels';
async function ready() {
  await page.waitForFunction(identifier=>{
    const api=window.__keplerComputeTest;
    const state=api.getState();
    if(state.error) throw new Error(state.error);
    return api.renderedLayers().some(layer=>layer.id===identifier && layer.layerClass==='GPUPointLabelLayer' && layer.loaded && layer.labelBufferIds.length===6);
  },layerId);
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
}
async function countText(name) {
  const image=PNG.sync.read(await page.screenshot({path:`/tmp/luma-kepler-point-labels-${name}.png`}));
  const centers=await page.evaluate(points=>points.map(([longitude,latitude])=>window.__keplerComputeTest.projectPosition(longitude,latitude)),positions);
  let count=0;
  for(const [centerX,centerY] of centers.slice(0,2))
    for(let positionY=Math.max(100,Math.floor(centerY-100));positionY<Math.min(image.height-100,centerY+100);positionY++)
      for(let positionX=Math.max(350,Math.floor(centerX-110));positionX<Math.min(image.width-80,centerX+110);positionX++) {
        const offset=(positionY*image.width+positionX)*4;
        if(image.data[offset]>170&&image.data[offset+1]>170&&image.data[offset+2]>170) count++;
      }
  return count;
}
try {
  await page.goto(process.argv[2]||'http://127.0.0.1:5318/?rows=256',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.__keplerComputeTest?.getState().selectedCount===256,undefined,{timeout:60000});
  await openNativeData(page,'Prepare data');
  const workspace=page.getByRole('region',{name:'Resident datasets',exact:true});
  await workspace.locator('input[type=file]').setInputFiles({name:'point-labels.arrow',mimeType:'application/vnd.apache.arrow.stream',buffer:Buffer.from(bytes)});
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  await workspace.getByRole('button',{name:'Check schema and precision',exact:true}).click();
  await workspace.getByRole('button',{name:'Publish selected fields',exact:true}).click();
  await page.waitForFunction(()=>window.__keplerComputeTest.getState().publications[0]?.selectedCount===3);
  await closeNativeDialog(page);
  datasetId=await page.evaluate(()=>window.__keplerComputeTest.getState().publications[0].id);
  const config=await page.evaluate(()=>window.__keplerComputeTest.saveNativeConfig());
  const layer=config.config.visState.layers.find(value=>value.config.dataId===datasetId);
  layer.id=layerId;
  layer.config.label='Resident point labels';
  layer.config.visConfig={...layer.config.visConfig,radius:10,filled:true,outline:false};
  layer.config.textLabel=[{field:null,color:[255,255,255],size:24,weight:400,anchor:'middle',alignment:'center',outlineWidth:0,background:false,collisionEnabled:false}];
  config.config.visState.layers=[layer];
  config.config.mapState={...config.config.mapState,longitude:-73.98,latitude:40.75,zoom:12,pitch:0,bearing:0};
  await page.evaluate(configuration=>window.__keplerComputeTest.applyNativeConfig(configuration),config);
  const labels=page.locator('.layer-config-group').filter({has:page.getByText('Label 1',{exact:true})}).first();
  if(!(await labels.isVisible())) await page.locator('.layer__enable-config svg').first().click();
  if ((await labels.getAttribute('class')).includes('collapsed'))
    await labels.locator('.layer-config-group__header').click();
  await labels.locator('.field-selector').last().click();
  await page.locator('.field-selector_list-item').filter({has:page.getByText('name',{exact:true})}).click();
  await page.keyboard.press('Escape');
  await ready();
  assert(await countText('native-field')>60,'Native label field must produce visible white glyphs');
  await page.evaluate(identifier=>{
    const resources=window.__keplerComputeTest.resourceIdentity();
    window.__pointLabelResources={labels:resources.labelBuffers,sources:resources.sourceBuffers[identifier]};
  },datasetId);
  const labelled=await page.evaluate(()=>window.__keplerComputeTest.saveNativeConfig());
  const active=labelled.config.visState.layers[0];
  active.config.visConfig.filled=false;
  active.config.visConfig.outline=false;
  active.config.textLabel[0]={...active.config.textLabel[0],size:30,anchor:'start',alignment:'top'};
  await page.evaluate(identifier=>{
    window.__keplerComputeTest.setPointMarkerFill(identifier,false);
    window.__keplerComputeTest.setPointLabelStyle(identifier,{size:30,anchor:'start',alignment:'top'});
  },layerId);
  await ready();
  assert(await countText('label-only')>60,'Labels remain visible when point markers are disabled');
  // Bitmap parity: the resident atlas is rasterized by the same 2D canvas path as Deck's font
  // atlas (Monaco, monospace). Compare ink extents of the first label with a same-page canvas
  // reference at the same pixel size; atlas filtering and hinting justify the tolerances.
  {
    const image = PNG.sync.read(await page.screenshot({path: '/tmp/luma-kepler-point-labels-parity.png'}));
    const [centerX, centerY] = await page.evaluate(([longitude, latitude]) => window.__keplerComputeTest.projectPosition(longitude, latitude), positions[0]);
    const bounds = {minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity};
    // Native 'start'/'top' labels sit up and to the right of their marker by the radius and padding,
    // so search a window that holds ALPHA but excludes BETA's label.
    for (let positionY = Math.max(100, Math.floor(centerY - 120)); positionY < Math.min(image.height - 100, centerY + 80); positionY++)
      for (let positionX = Math.max(350, Math.floor(centerX - 20)); positionX < Math.min(image.width - 80, centerX + 160); positionX++) {
        const offset = (positionY * image.width + positionX) * 4;
        if (image.data[offset] > 170 && image.data[offset + 1] > 170 && image.data[offset + 2] > 170) {
          bounds.minX = Math.min(bounds.minX, positionX); bounds.maxX = Math.max(bounds.maxX, positionX);
          bounds.minY = Math.min(bounds.minY, positionY); bounds.maxY = Math.max(bounds.maxY, positionY);
        }
      }
    assert(bounds.maxX > bounds.minX, 'Resident label ink must be present for bitmap parity');
    const reference = await page.evaluate(({text, size, weight}) => {
      const canvas = document.createElement('canvas');
      canvas.width = 400; canvas.height = 120;
      const context = canvas.getContext('2d', {willReadFrequently: true});
      context.fillStyle = '#000'; context.fillRect(0, 0, 400, 120);
      context.fillStyle = '#fff'; context.textBaseline = 'alphabetic';
      context.font = `${weight} ${size}px Monaco, monospace`;
      context.fillText(text, 20, 80);
      const data = context.getImageData(0, 0, 400, 120).data;
      const box = {minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity};
      for (let y = 0; y < 120; y++) for (let x = 0; x < 400; x++) {
        const offset = (y * 400 + x) * 4;
        if (data[offset] > 170 && data[offset + 1] > 170 && data[offset + 2] > 170) {
          box.minX = Math.min(box.minX, x); box.maxX = Math.max(box.maxX, x);
          box.minY = Math.min(box.minY, y); box.maxY = Math.max(box.maxY, y);
        }
      }
      return box;
    }, {text: 'ALPHA', size: 30 * (2 / 3), weight: 400}); // Kepler renders label size at PROJECTED_PIXEL_SIZE_MULTIPLIER (2/3)
    const width = bounds.maxX - bounds.minX, height = bounds.maxY - bounds.minY;
    const referenceWidth = reference.maxX - reference.minX, referenceHeight = reference.maxY - reference.minY;
    assert(Math.abs(width - referenceWidth) <= Math.max(6, referenceWidth * 0.12), `Label ink width ${width} must match canvas bitmap reference ${referenceWidth}`);
    assert(Math.abs(height - referenceHeight) <= Math.max(6, referenceHeight * 0.2), `Label ink height ${height} must match canvas bitmap reference ${referenceHeight}`);
  }
  assert(await page.evaluate(()=>Object.entries(window.__pointLabelResources.labels).every(([id,buffers])=>window.__keplerComputeTest.resourceIdentity().labelBuffers[id]?.every((buffer,index)=>buffer===buffers[index]))),'Font layout buffers must survive display-only controls');
  await page.evaluate(identifier=>window.__keplerComputeTest.setPublishedRange(identifier,'value',3,3),datasetId);
  await page.waitForFunction(identifier=>window.__keplerComputeTest.getState().publications.find(value=>value.id===identifier)?.selectedCount===1,datasetId);
  await ready();
  assert.equal(await countText('null-only'),0,'Selected null source row must not acquire another row label');
  await page.evaluate(configuration=>window.__keplerComputeTest.applyNativeConfig(configuration),labelled);
  await ready();
  assert(await countText('restored')>60);
  await page.evaluate(()=>{window.__pointLabelResources.labels=window.__keplerComputeTest.resourceIdentity().labelBuffers;});
  const numeric=structuredClone(labelled);
  numeric.config.visState.layers[0].config.textLabel[0].field={name:'value',type:'real'};
  await page.evaluate(configuration=>window.__keplerComputeTest.applyNativeConfig(configuration),numeric);
  await ready();
  await page.waitForFunction(()=>Object.values(window.__pointLabelResources.labels).flat().every(buffer=>buffer.destroyed));
  assert(await page.evaluate(identifier=>window.__keplerComputeTest.resourceIdentity().sourceBuffers[identifier].every((buffer,index)=>buffer===window.__pointLabelResources.sources[index]),datasetId));
  const removed=structuredClone(labelled);
  removed.config.visState.layers[0].config.textLabel=[];
  await page.evaluate(configuration=>window.__keplerComputeTest.applyNativeConfig(configuration),removed);
  await page.waitForFunction(()=>!window.__keplerComputeTest.renderedLayers().some(layer=>layer.layerClass==='GPUPointLabelLayer'));
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,sourceBatches:[2,0,1],checks:['native scalar label dropdown and visible glyph pixels','label-only rendering and display controls reuse glyph buffers','ALPHA ink extents match a same-page canvas bitmap reference at the Deck atlas font and pixel size','source-aligned null/filter behavior','field replacement retires glyph resources behind submitted frame','original source buffers retained and labels removed']},null,2));
} catch(error) {
  console.error(JSON.stringify(await page.evaluate(()=>({state:window.__keplerComputeTest?.getState(),layers:window.__keplerComputeTest?.renderedLayers()})).catch(()=>null),null,2));
  console.error(errors);
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
