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
  name: arrow.vectorFromArray(indexes.map(index => ['ALPHA\nbeta','BETA',null][index]), new arrow.Utf8())
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
const layerId='native-label-styles';
async function ready() {
  await page.waitForFunction(identifier=>{
    const api=window.__keplerComputeTest;
    if(api.getState().error) throw new Error(api.getState().error);
    return api.renderedLayers().some(layer=>layer.id===identifier && layer.layerClass==='GPUPointLabelLayer' && layer.loaded);
  },layerId);
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
}
// Counts pixels near the first two labels that satisfy a color predicate and reports their ink extent.
async function measure(name,predicate) {
  const image=PNG.sync.read(await page.screenshot({path:`/tmp/luma-kepler-${name}.png`}));
  const [centerX,centerY]=await page.evaluate(([longitude,latitude])=>window.__keplerComputeTest.projectPosition(longitude,latitude),positions[0]);
  const box={count:0,minY:Infinity,maxY:-Infinity};
  for(let y=Math.max(100,Math.floor(centerY-160));y<Math.min(image.height-100,centerY+160);y++)
    for(let x=Math.max(350,Math.floor(centerX-40));x<Math.min(image.width-80,centerX+200);x++){
      const offset=(y*image.width+x)*4;
      if(predicate(image.data[offset],image.data[offset+1],image.data[offset+2])){box.count++;box.minY=Math.min(box.minY,y);box.maxY=Math.max(box.maxY,y);}
    }
  return box;
}
const isWhite=(r,g,b)=>r>170&&g>170&&b>170;
const isOutline=(r,g,b)=>r>200&&g<90&&b<90;
const isBackground=(r,g,b)=>b>160&&r<60&&g<60;
try {
  await page.goto(process.argv[2]||'http://127.0.0.1:5318/?rows=256',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.__keplerComputeTest?.getState().selectedCount===256,undefined,{timeout:60000});
  await openNativeData(page,'Prepare data');
  const workspace=page.getByRole('region',{name:'Resident datasets',exact:true});
  await workspace.locator('input[type=file]').setInputFiles({name:'label-styles.arrow',mimeType:'application/vnd.apache.arrow.stream',buffer:Buffer.from(bytes)});
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  await workspace.getByRole('button',{name:'Check schema and precision',exact:true}).click();
  await workspace.getByRole('button',{name:'Publish selected fields',exact:true}).click();
  await page.waitForFunction(()=>window.__keplerComputeTest.getState().publications[0]?.selectedCount===3);
  await closeNativeDialog(page);
  datasetId=await page.evaluate(()=>window.__keplerComputeTest.getState().publications[0].id);
  const base=await page.evaluate(()=>window.__keplerComputeTest.saveNativeConfig());
  const apply=async textLabel=>{
    const config=structuredClone(base);
    const layer=config.config.visState.layers.find(value=>value.config.dataId===datasetId);
    layer.id=layerId;
    layer.config.visConfig={...layer.config.visConfig,radius:10,filled:true,outline:false};
    layer.config.textLabel=[{field:{name:'name',type:'string'},color:[255,255,255],size:24,weight:400,anchor:'middle',alignment:'center',outlineWidth:0,outlineColor:[255,0,0],background:false,backgroundColor:[0,0,255,255],collisionEnabled:false,...textLabel}];
    config.config.visState.layers=[layer];
    config.config.mapState={...config.config.mapState,longitude:-73.98,latitude:40.75,zoom:12,pitch:0,bearing:0};
    await page.evaluate(configuration=>window.__keplerComputeTest.applyNativeConfig(configuration),config);
    await ready();
  };
  await apply({});
  const plain=await measure('label-styles-plain',isWhite);
  assert(plain.count>60,'Multiline label draws glyphs');
  assert(plain.maxY-plain.minY>24*(2/3)*1.5,`Two-line label must be taller than one line (${plain.maxY-plain.minY})`);
  assert.equal((await measure('label-styles-no-outline',isOutline)).count,0,'No outline without outlineWidth');
  assert.equal((await measure('label-styles-no-background',isBackground)).count,0,'No background by default');
  await apply({outlineWidth:0.1});
  assert((await measure('label-styles-outline',isOutline)).count>40,'SDF outline must draw outline-colored pixels around the fill');
  await apply({background:true});
  assert((await measure('label-styles-background',isBackground)).count>200,'Background rectangle must draw behind the label');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,checks:['multiline height','SDF outline pixels','row background pixels']},null,2));
} catch(error) {
  console.error(JSON.stringify(await page.evaluate(()=>({state:window.__keplerComputeTest?.getState(),layers:window.__keplerComputeTest?.renderedLayers()})).catch(()=>null),null,2));
  console.error(errors);
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
