import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const oracle=JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json',import.meta.url),'utf8')).spectralDisplay;
const baseUrl=process.argv[2]??'http://127.0.0.1:5313/';
const outputDirectory=process.argv[3]??'/tmp/luma-showcase-overhaul-20261025/display-numerics';
await mkdir(outputDirectory,{recursive:true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const browser=await chromium.launch(getPlaywrightLaunchOptions({headless:true,backend:'webgpu-core'}));
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const results=[],errors=[];
page.on('pageerror',error=>errors.push(String(error)));
page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
const select=(id,value)=>page.locator(`#panel-controls select[id$="option-${id}"]`).selectOption(String(value));
const readout=id=>page.locator(`[data-readout="${id}"]`).first().textContent();
const ready=()=>page.waitForFunction(()=>document.querySelector('[data-readout="transportEvidence"]')?.textContent?.includes('finite comparisons'),null,{timeout:180000});
const closeTo=(actual,expected,tolerance,label)=>assert.ok(Number.isFinite(actual)&&Math.abs(actual-expected)<=tolerance,`${label}: ${actual}, expected ${expected} ± ${tolerance}`);
try{
 await page.goto(`${baseUrl.replace(/\/$/,'')}/#/story/spectral-texture/observation`,{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>globalThis.spatialAnalysisShowcase?.ready||globalThis.spatialAnalysisShowcase?.error,null,{timeout:180000});
 assert.equal(await page.evaluate(()=>globalThis.spatialAnalysisShowcase?.error),null);
 await installReadoutIdentifiers(page, 'spectral-texture');
 await page.locator('#tab-controls').click();await ready();
 for(const reference of oracle.display){
  for(const [id,value]of Object.entries({date:reference.date,displayDomain:reference.domain,displayMode:reference.mode,displayContrast:reference.contrast}))await select(id,value);
  if(reference.mode !== 'linear') await select('displayGamma', reference.gamma);
  if(reference.mode === 'equalize') await select('displayBins', reference.bins);
  await ready();const text=await readout('displayEvidence');const match=text.match(/→ display ([\d.]+) · ([\d,]+) invalid/);assert.ok(match,text);closeTo(Number(match[1]),reference.probe,.00002,'Display value');assert.equal(Number(match[2].replaceAll(',','')),reference.invalid);results.push({display:true,...reference,text});
 }
 for(const reference of oracle.profiles){
  for(const [id,value]of Object.entries({date:reference.date,profileSpacing:reference.spacing,profileMethod:reference.method}))await select(id,value);
  if(reference.method === 'bilinear') await select('profileNoData',reference.policy);
  await ready();const text=await readout('profileEvidence');const match=text.match(/(\d+)\/(\d+) emitted · (\d+) finite · overflow (\d+) · .* · min ([\d.]+) \/ max ([\d.]+) · ([\d.]+) m/);assert.ok(match,text);assert.equal(Number(match[1]),reference.count);assert.equal(Number(match[2]),reference.count);assert.equal(Number(match[3]),reference.finite);assert.equal(Number(match[4]),0);closeTo(Number(match[5]),reference.minimum,.00003,'Profile minimum');closeTo(Number(match[6]),reference.maximum,.00003,'Profile maximum');closeTo(Number(match[7]),reference.length,.06,'Profile length');results.push({profile:true,...reference,effectivePolicy:reference.method === 'nearest' ? 'not-used-by-nearest' : reference.policy,text});
 }
 for(const reference of oracle.population){
  await select('date',reference.date);
  for(const channel of ['R','G','B','A']){
   await select('transportChannel',channel);await ready();const text=await readout('transportEvidence');const match=text.match(/returned\| ([\d.e+-]+) · validity mismatches (\d+) · (\d+) finite/);assert.ok(match,text);assert.equal(Number(match[1]),0);assert.equal(Number(match[2]),0);assert.equal(Number(match[3]),reference.count);results.push({transport:true,date:reference.date,channel,text});
  }
 }
 assert.deepEqual(errors,[]);await page.locator('#tab-story').click();await page.evaluate(()=>{window.location.hash='#/story/spectral-texture/transect';});await ready();await page.screenshot({path:`${outputDirectory}/transect.png`,fullPage:true});
 await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:true,results,errors},null,2));console.log(JSON.stringify({passed:true,checks:results.length,errors}));
}catch(error){await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:false,error:String(error),results,errors},null,2));throw error;}
finally{await browser.close(); await verifyBuildEvidence();}
