import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const source=JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json',import.meta.url),'utf8')).gridAlignment;
const baseUrl=process.argv[2]??'http://127.0.0.1:5313/';
const outputDirectory=process.argv[3]??'/tmp/luma-showcase-overhaul-20261025/grid-numerics';
await mkdir(outputDirectory,{recursive:true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const browser=await chromium.launch(getPlaywrightLaunchOptions({headless:true,backend:'webgpu-core'}));
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[],results=[];
await page.addInitScript(() => {window.addEventListener('grid-alignment-diagnostic', event => {globalThis.gridAlignmentDiagnostic = event.detail;});});
page.on('pageerror',error=>errors.push(String(error)));
page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
const select=(id,value)=>page.locator(`#panel-controls select[id$="option-${id}"]`).selectOption(value);
const readout=id=>page.locator(`[data-readout="${id}"]`).first().textContent();
const closeTo=(actual,expected,tolerance,label)=>assert.ok(Number.isFinite(actual)&&Math.abs(actual-expected)<=tolerance,`${label}: ${actual}, expected ${expected} ± ${tolerance}`);
try{
 await page.goto(`${baseUrl.replace(/\/$/,'')}/?gridSupportDiagnostics#/story/grid-alignment/source-grid`,{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>globalThis.spatialAnalysisShowcase?.ready||globalThis.spatialAnalysisShowcase?.error,null,{timeout:180000});
 assert.equal(await page.evaluate(()=>globalThis.spatialAnalysisShowcase?.error),null);
 await installReadoutIdentifiers(page, 'grid-alignment');
 await page.locator('#tab-controls').click();
 for(const mode of ['periodic','masked-linear']){
  await select('experiment',mode);await select('product','difference');
  await page.waitForFunction(()=>document.querySelector('[data-readout="registrationResult"]')?.textContent?.includes(' · accepted ·'),null,{timeout:180000});
  const text=await readout('registrationResult');const match=text.match(/GPU correction \[(-?[\d.]+), (-?[\d.]+)\]/);assert.ok(match,text);
  const expected=mode==='periodic'?source.periodicCorrection:source.maskedCorrection;
  closeTo(Number(match[1]),expected[0],.01,`${mode} column`);closeTo(Number(match[2]),expected[1],.01,`${mode} row`);
  const difference=await readout('alignmentDifference');const statistics=difference.match(/([\d,]+) valid aligned cells · max \|difference\| ([\d.]+) · RMS ([\d.]+)/);assert.ok(statistics,difference);
  assert.equal(Number(statistics[1].replaceAll(',','')),mode==='periodic'?source.periodicAlignedCount:source.maskedAlignedCount);
  closeTo(Number(statistics[2]),0,.00002,`${mode} maximum alignment residual`);closeTo(Number(statistics[3]),0,.00001,`${mode} RMS alignment residual`);results.push({mode,text,difference});
 }
 await select('product','reprojected');
 for(const reference of source.records){
  await select('reprojectDate',reference.date);await select('method',reference.method);await select('noDataPolicy',reference.policy);
  const text=await readout('reprojectionSupport');const match=text.match(/([\d,]+) \/ 4,096 destination cells valid.*selected cell ([\d.]+)/);assert.ok(match,text);
  const count=Number(match[1].replaceAll(',',''));const value=Number(match[2]);
  // Exact support-count agreement is deliberately audited separately: f32 CRS
  // projection can move an interpolation footprint across a validity boundary.
  closeTo(value,reference.probe,.002,`${reference.date} ${reference.method} selected reflectance`);
  results.push({reprojection:true,date:reference.date,method:reference.method,policy:reference.policy,count,referenceCount:reference.validCount,countDifference:count-reference.validCount,selected:value,referenceSelected:reference.probe,text});
 }
 for(const date of ['before','after']){
  const subset=results.filter(result=>result.reprojection&&result.date===date);
  const item=(method,policy)=>subset.find(result=>result.method===method&&result.policy===policy);
  assert.equal(item('nearest','strict').count,item('nearest','renormalize').count);
  assert.ok(item('bilinear','strict').count<=item('bilinear','renormalize').count);
 }
 assert.deepEqual(errors,[]);
 const diagnostic = await page.evaluate(() => globalThis.gridAlignmentDiagnostic ?? null);
 if (diagnostic) await writeFile(`${outputDirectory}/gpu-support.json`, JSON.stringify(diagnostic));
 await page.locator('#tab-story').click();await page.evaluate(()=>{window.location.hash='#/story/grid-alignment/nodata-support';});
 await page.screenshot({path:`${outputDirectory}/grid-support.png`,fullPage:true});
 const projectionCountAuditPassed=results.filter(result=>result.reprojection).every(result=>result.countDifference===0);
 await writeFile(`${outputDirectory}/report.json`,JSON.stringify({registrationPassed:true,probeAndSupportPolicyPassed:true,projectionCountAuditPassed,results,errors},null,2));
 console.log(JSON.stringify({registrationPassed:true,probeAndSupportPolicyPassed:true,projectionCountAuditPassed,results,errors},null,2));
}catch(error){await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:false,error:String(error),results,errors},null,2));throw error;}
finally{await browser.close(); await verifyBuildEvidence();}
