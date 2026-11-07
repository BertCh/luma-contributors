import {captureBuildEvidence} from './earth-build-evidence.mjs';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';
const oracle=JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json',import.meta.url),'utf8')).surfacePreparation;
const baseUrl=process.argv[2]??'http://127.0.0.1:5313/';
const outputDirectory=process.argv[3]??'/tmp/luma-showcase-overhaul-20261025/preparation-numerics';
await mkdir(outputDirectory,{recursive:true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const browser=await chromium.launch(getPlaywrightLaunchOptions({headless:true,backend:'webgpu-core'}));
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const results=[],errors=[];
page.on('pageerror',error=>errors.push(String(error)));
page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
const select=(id,value)=>page.locator(`#panel-controls select[id$="option-${id}"]`).selectOption(String(value));
const readout=id=>page.locator(`[data-readout="${id}"]`).first().textContent();
const ready=()=>page.waitForFunction(()=>document.querySelector('[data-readout="preparationEvidence"]')?.textContent?.includes('unresolved after preparation'),null,{timeout:180000});
const closeTo=(actual,expected,tolerance,label)=>assert.ok(Number.isFinite(actual)&&Math.abs(actual-expected)<=tolerance,`${label}: ${actual}, expected ${expected} ± ${tolerance}`);
try{
 await page.goto(`${baseUrl.replace(/\/$/,'')}/#/story/missing-observations/bounded-fill`,{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>globalThis.spatialAnalysisShowcase?.ready||globalThis.spatialAnalysisShowcase?.error,null,{timeout:180000});assert.equal(await page.evaluate(()=>globalThis.spatialAnalysisShowcase?.error),null);await installReadoutIdentifiers(page,'missing-observations');await page.locator('#tab-controls').click();await ready();await select('fillMethod','nearest');await select('preparationFilter','median');
 for(const reference of oracle.records){
  await select('experiment',reference.scenario==='september'?'september-gaps':'held-out');if(reference.scenario!=='september')await select('holdoutPattern',reference.scenario);await select('maximumDistance',reference.distance);await select('preparationRadius',reference.radius);await select('preparationNoData',reference.policy);await select('preparationBorderMode',reference.borderMode);await ready();
  const text=await readout('preparationEvidence');const match=text.match(/([\d,]+) original valid · ([\d,]+) filled · ([\d,]+) rank-filter changed · ([\d,]+) unresolved/);assert.ok(match,text);for(const [position,key]of [[1,'original'],[2,'filled'],[3,'filtered'],[4,'unresolved']])assert.equal(Number(match[position].replaceAll(',','')),reference[key],key);assert.ok(text.includes(`${reference.policy} invalid-neighbor policy · ${reference.borderMode} border`),text);
  if(reference.scenario!=='september'&&reference.evaluated){const metrics=text.match(/RMSE ([\d.]+) \/ MAE ([\d.]+)/);assert.ok(metrics,text);closeTo(Number(metrics[1]),reference.rmse,.00006,'Prepared withheld RMSE');closeTo(Number(metrics[2]),reference.mae,.00006,'Prepared withheld MAE');}
  const selected=await readout('preparationPixel');if(reference.probe===null)assert.match(selected,/Prepared unresolved/);else{const value=selected.match(/Prepared ([\d.]+)/);assert.ok(value,selected);closeTo(Number(value[1]),reference.probe,.00006,'Prepared selected value');}results.push({...reference,text,selected});console.log(`${results.length}/${oracle.records.length} ${reference.scenario}/${reference.distance}/${reference.radius}/${reference.policy}`);
 }
 assert.deepEqual(errors,[]);await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:true,results,errors},null,2));console.log(JSON.stringify({passed:true,checks:results.length,errors}));
}catch(error){await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:false,error:String(error),results,errors},null,2));throw error;}
finally{await browser.close(); await verifyBuildEvidence();}
