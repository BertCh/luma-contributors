import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const oracle=JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json',import.meta.url),'utf8')).tiledObservations;
const baseUrl=process.argv[2]??'http://127.0.0.1:5313/';
const outputDirectory=process.argv[3]??'/tmp/luma-showcase-overhaul-20261025/tile-numerics';
await mkdir(outputDirectory,{recursive:true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const browser=await chromium.launch(getPlaywrightLaunchOptions({headless:true,backend:'webgpu-core'}));
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const results=[],errors=[];
page.on('pageerror',error=>errors.push(String(error)));
page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
const select=(id,value)=>page.locator(`#panel-controls select[id$="option-${id}"]`).selectOption(String(value));
const readout=id=>page.locator(`[data-readout="${id}"]`).first().textContent();
const ready=()=>page.waitForFunction(()=>document.querySelector('[data-readout="overviewAgreement"]')?.textContent?.includes('counts'),null,{timeout:180000});
const closeTo=(actual,expected,tolerance,label)=>assert.ok(Number.isFinite(actual)&&Math.abs(actual-expected)<=tolerance,`${label}: ${actual}, expected ${expected} ± ${tolerance}`);
const setChecked = async (locator, expected) => {
  if (await locator.isChecked() !== expected) {
    await locator.focus();
    await locator.press('Space');
  }
  assert.equal(await locator.isChecked(), expected);
};

try{
 await page.goto(`${baseUrl.replace(/\/$/,'')}/#/story/tiled-observations/ownership`,{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>globalThis.spatialAnalysisShowcase?.ready||globalThis.spatialAnalysisShowcase?.error,null,{timeout:180000});assert.equal(await page.evaluate(()=>globalThis.spatialAnalysisShowcase?.error),null);
 await installReadoutIdentifiers(page, 'tiled-observations');await page.locator('#tab-controls').click();await ready();
 for(const reference of oracle.records){
  await select('date',reference.date==='before'?'july':'september');await ready();
  assert.equal(Number((await readout('validCells')).replaceAll(',','')),reference.count);
  const sumText=await readout('sourceSumCheck');const sum=sumText.match(/([\d.]+) GPU/);assert.ok(sum,sumText);closeTo(Number(sum[1]),reference.sum,.003,'Owned-source sum');results.push({population:true,date:reference.date,sumText});
  for(const histogram of reference.histograms){
   await select('binCount',histogram.bins);
   for(const [probability,expected]of Object.entries(histogram.quantiles)){
    await select('percentile',probability);await ready();const text=await readout('percentile');const match=text.match(/≈ ([\d.]+)/);assert.ok(match,text);closeTo(Number(match[1]),expected,.00051,'Histogram quantile');const details=await readout('histogramDetails');assert.match(details,new RegExp(`counted ${reference.count} · overflow 0`));results.push({histogram:true,date:reference.date,bins:histogram.bins,probability,text,details});
   }
  }
  for(const testCase of reference.cases){
   await select('edgePolicy',testCase.border);await select('noDataPolicy',testCase.policy);await ready();
   const seam=await readout('wholecropSeamError');const match=seam.match(/(\d+) shared-valid cells.*max ([\d.]+) · RMS ([\d.]+)/);assert.ok(match,seam);assert.equal(Number(match[1]),testCase.validBlur);closeTo(Number(match[2]),0,.000002,'Tiled/wholecrop maximum');closeTo(Number(match[3]),0,.000002,'Tiled/wholecrop RMS');
   const agreement=await readout('overviewAgreement');assert.match(agreement,/counts agree exactly/);const deltas=agreement.match(/Δsum\| ([\d.e+-]+) · max \|Δmean\| ([\d.e+-]+)/);assert.ok(deltas,agreement);closeTo(Number(deltas[1]),0,.00002,'Overview sum conservation');closeTo(Number(deltas[2]),0,.000002,'Overview mean conservation');
   for(const category of testCase.categorical){
    await select('categoricalPolicy',category.policy);await ready();const text=await readout('categoricalOverview');const match=text.match(/(\d+) valid labels · support (\d+) · bins 0:(\d+) 1:(\d+) 2:(\d+) 3:(\d+)/);assert.ok(match,text);assert.equal(Number(match[1]),category.valid);assert.equal(Number(match[2]),category.support);assert.deepEqual(match.slice(3).map(Number),category.classes);results.push({category:true,date:reference.date,...testCase,category,text,seam,agreement});
   }
  }
  for(const omitEast of [false,true]){
   await setChecked(page.locator('#panel-controls input[id$="option-omitEast"]'), omitEast);
   for(const resample of ['nearest','bilinear-strict','bilinear-renormalize']){
    await select('resampling',resample);await ready();const text=await readout('resampleSupport');const match=text.match(/(\d+) tiled destinations · (\d+) reference destinations · (\d+) support mismatches · (\d+) shared support · max \|Δ\| ([\d.e+-]+)/);assert.ok(match,text);assert.ok(Number(match[1])>0);assert.equal(match[1],match[2]);assert.equal(Number(match[3]),0);assert.equal(match[1],match[4]);closeTo(Number(match[5]),0,.000002,'Tiled/untiled resampling');results.push({resampling:true,date:reference.date,omitEast,resample,text});
   }
  }
  await setChecked(page.locator('#panel-controls input[id$="option-omitEast"]'), false);
 }
 assert.deepEqual(errors,[]);await page.locator('#tab-story').click();await page.evaluate(()=>{window.location.hash='#/story/tiled-observations/seams';});await ready();await page.screenshot({path:`${outputDirectory}/tile-seams.png`,fullPage:true});await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:true,results,errors},null,2));console.log(JSON.stringify({passed:true,checks:results.length,errors}));
}catch(error){await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:false,error:String(error),results,errors},null,2));throw error;}
finally{await browser.close(); await verifyBuildEvidence();}
