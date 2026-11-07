import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const oracle=JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json',import.meta.url),'utf8')).missingObservations;
const baseUrl=process.argv[2]??'http://127.0.0.1:5313/';
const outputDirectory=process.argv[3]??'/tmp/luma-showcase-overhaul-20261025/missing-numerics';
await mkdir(outputDirectory,{recursive:true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const browser=await chromium.launch(getPlaywrightLaunchOptions({headless:true,backend:'webgpu-core'}));
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const results=[],errors=[];
page.on('pageerror',error=>errors.push(String(error)));
page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
const select=(id,value)=>page.locator(`#panel-controls select[id$="option-${id}"]`).selectOption(String(value));
const readout=id=>page.locator(`[data-readout="${id}"]`).first().textContent();
const ready=()=>page.waitForFunction(()=>document.querySelector('[data-readout="fillSupport"]')?.textContent?.includes('unresolved within'),null,{timeout:180000});
const closeTo=(actual,expected,tolerance,label)=>assert.ok(Number.isFinite(actual)&&Math.abs(actual-expected)<=tolerance,`${label}: ${actual}, expected ${expected} ± ${tolerance}`);
try{
 await page.goto(`${baseUrl.replace(/\/$/,'')}/#/story/missing-observations/measured-support`,{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>globalThis.spatialAnalysisShowcase?.ready||globalThis.spatialAnalysisShowcase?.error,null,{timeout:180000});
 assert.equal(await page.evaluate(()=>globalThis.spatialAnalysisShowcase?.error),null);
 await installReadoutIdentifiers(page, 'missing-observations');
 await page.locator('#tab-controls').click();await ready();
 const records=oracle.fills.filter(record=>process.argv.includes('--all')||record.radius===8||record.scenario==='block');
 for(const reference of records){
  await select('experiment',reference.scenario==='september'?'september-gaps':'held-out');
  if(reference.scenario!=='september')await select('holdoutPattern',reference.scenario);
  await select('maximumDistance',reference.radius);await select('fillMethod',reference.method);await ready();
  const support=await readout('fillSupport');const match=support.match(/([\d,]+) filled-surface cells valid · ([\d,]+) cells interpolated · ([\d,]+) unresolved/);assert.ok(match,support);
  for(const [index,key]of [[1,'validCount'],[2,'filledCount'],[3,'unresolved']])assert.equal(Number(match[index].replaceAll(',','')),reference[key],`${reference.scenario}/${reference.radius}/${reference.method} ${key}`);
  const errorText=await readout('holdoutError');
  if(reference.scenario==='september')assert.match(errorText,/No withheld truth/);
  else{
   const error=errorText.match(/([\d,]+) \/ ([\d,]+) withheld cells evaluated · RMSE ([\d.]+) · MAE ([\d.]+)/);assert.ok(error,errorText);
   assert.equal(Number(error[1].replaceAll(',','')),reference.filledCount);assert.equal(Number(error[2].replaceAll(',','')),reference.heldoutCount);
   closeTo(Number(error[3]),reference.rmse,.00006,'Withheld RMSE');closeTo(Number(error[4]),reference.mae,.00006,'Withheld MAE');
  }
  const selected=await readout('selectedPixel');
  if(reference.probe===null)assert.match(selected,/unresolved$/);
  else{const probe=selected.match(/ · (?:nearest|inverse-distance|gdal-four-quadrant) ([\d.]+)$/);assert.ok(probe,selected);closeTo(Number(probe[1]),reference.probe,.00006,'Selected prediction');}
  results.push({...reference,support,errorText,selected});
 }
 await select('experiment','held-out');
 for(const reference of oracle.morphology){
  await select('morphologyRadius',reference.radius);await select('structuringElement',reference.element);await ready();
  const chart=await page.locator('[data-readout-value="structureChart"] table').first().evaluate(table=>Object.fromEntries(Array.from(table.querySelectorAll('tbody tr'),row=>{const cells=Array.from(row.querySelectorAll('td'),cell=>cell.textContent.trim());return [cells[0],Number(cells[1])];})));
  for(const [operation,expected]of Object.entries(reference.probe))closeTo(chart[operation],expected,.00051,`${reference.element}/${reference.radius} ${operation} chart`);
  results.push({morphology:true,...reference,chart});
 }
 for(const reference of oracle.holes){
  await select('experiment',reference.scenario==='september'?'september-gaps':'held-out');
  if(reference.scenario!=='september')await select('holdoutPattern',reference.scenario);
  await select('holeConnectivity',reference.connectivity);await ready();
  const text=await readout('holeFillStatus');const match=text.match(/([\d,]+) enclosed missing cells/);assert.ok(match,text);assert.equal(Number(match[1].replaceAll(',','')),reference.filledCount);results.push({hole:true,...reference,text});
 }
 await select('experiment','held-out');await select('maximumIterations','128');
 for(const reference of oracle.reconstruction){
  await select('reconstructionMode',reference.mode);await select('reconstructionConnectivity',reference.connectivity);await ready();
  const text=await readout('reconstructionStatus');assert.match(text,/converged within 128 rounds/);assert.ok(!text.includes('not converged'));results.push({reconstruction:true,...reference,text});
 }
 for(const reference of oracle.contours){
  const control=page.locator('#panel-controls input[id$="option-contourLevel"]');
  await control.fill(String(reference.level));await control.dispatchEvent('input');await control.dispatchEvent('change');await ready();
  const text=await readout('contourEvidence');const match=text.match(/([\d,]+) of ([\d,]+) required segments · ([\d,]+) cells cross the level · overflow no/);assert.ok(match,text);assert.equal(Number(match[1].replaceAll(',','')),reference.segments);assert.equal(Number(match[2].replaceAll(',','')),reference.segments);assert.equal(Number(match[3].replaceAll(',','')),reference.crossingCells);results.push({contours:true,...reference,text});
 }
 assert.deepEqual(errors,[]);
 await page.locator('#tab-story').click();await page.evaluate(()=>{window.location.hash='#/story/missing-observations/known-truth';});await ready();
 await page.screenshot({path:`${outputDirectory}/withheld-truth.png`,fullPage:true});
 await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:true,results,errors},null,2));console.log(JSON.stringify({passed:true,checks:results.length,errors}));
}catch(error){await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:false,error:String(error),results,errors},null,2));throw error;}
finally{await browser.close(); await verifyBuildEvidence();}
