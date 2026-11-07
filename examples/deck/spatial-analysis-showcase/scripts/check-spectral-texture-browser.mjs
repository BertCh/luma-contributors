import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const oracles = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json', import.meta.url), 'utf8'));
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/texture-numerics';
await mkdir(outputDirectory, {recursive:true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless:true,backend:'webgpu-core'}));
const page = await browser.newPage({viewport:{width:1440,height:1000}});
const results=[],errors=[];
page.on('pageerror',error=>errors.push(String(error)));
page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
const select=(id,value)=>page.locator(`#panel-controls select[id$="option-${id}"]`).selectOption(value);
const readout=id=>page.locator(`[data-readout="${id}"]`).first().textContent();
const closeTo=(actual,expected,tolerance,label)=>assert.ok(Number.isFinite(actual)&&Math.abs(actual-expected)<=tolerance,`${label}: ${actual}, expected ${expected} ± ${tolerance}`);
const ready=()=>page.waitForFunction(()=>document.querySelector('[data-readout="focalEvidence"]')?.textContent?.includes('status mask'),null,{timeout:180000});
try {
  await page.goto(`${baseUrl.replace(/\/$/,'')}/#/story/spectral-texture/observation`,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>globalThis.spatialAnalysisShowcase?.ready||globalThis.spatialAnalysisShowcase?.error,null,{timeout:180000});
  assert.equal(await page.evaluate(()=>globalThis.spatialAnalysisShowcase?.error),null);
 await installReadoutIdentifiers(page, 'spectral-texture');
  await page.locator('#tab-controls').click();
  for(const reference of oracles.spectralTextureBothDates.records){
    await select('layer','convolution');
    await select('convolutionKernel','sobel-x');
    await select('date',reference.date);
    await ready();
    for(const [layer,expected] of Object.entries(reference.layers)){
      await select('layer',layer);
      const text=await readout('selectedEvidence');
      const match=text.match(new RegExp(` · ${layer} (-?[\\d.]+) ·`));
      assert.ok(match,text);closeTo(Number(match[1]),expected,0.000003,`${reference.date} ${layer}`);
      results.push({date:reference.date,layer,text});
    }
    await select('layer','convolution');
    for(const [kernel,expected] of Object.entries(reference.rectangularConvolution)){
      await select('convolutionKernel',kernel);await ready();
      const text=await readout('selectedEvidence');const match=text.match(/ · convolution (-?[\d.]+) ·/);
      assert.ok(match,text);closeTo(Number(match[1]),expected,0.000003,`${reference.date} ${kernel}`);results.push({date:reference.date,kernel,text});
    }
    await select('layer','texture');
    for(const [metric,expected] of Object.entries(reference.glcm).filter(([name])=>name!=='pairCount')){
      await select('metric',metric);
      const text=await readout('textureEvidence');const match=text.match(new RegExp(`^${metric} (-?[\\d.]+) · (\\d+) directed pairs`));
      assert.ok(match,text);closeTo(Number(match[1]),expected,0.00001,`${reference.date} ${metric}`);assert.equal(Number(match[2]),reference.glcm.pairCount);results.push({date:reference.date,metric,text});
    }
    await select('layer','focal');
    await select('covariance','sample');await ready();
    const text=await readout('focalEvidence');
    for(const [label,key] of [['cov','sampleCovariance'],['r','correlation'],['slope','slope'],['intercept','intercept'],['R²','rSquared'],['RSS','residualSumSquares']]){
      const match=text.match(new RegExp(`(?:^| · )${label} (-?[\\d.]+)`));assert.ok(match,text);closeTo(Number(match[1]),reference.focal[key],0.00001,`${reference.date} ${label}`);
    }
    assert.match(text,/n 49 · status mask 0: supported/);results.push({date:reference.date,focal:text});
    await select('covariance','population');await ready();
    const population=await readout('focalEvidence');const match=population.match(/^cov (-?[\d.]+)/);assert.ok(match,population);closeTo(Number(match[1]),reference.focal.populationCovariance,0.000001,`${reference.date} population covariance`);
    await select('covariance','sample');await ready();
  }
  assert.deepEqual(errors,[]);
  await page.locator('#tab-story').click();
  await page.evaluate(()=>{window.location.hash='#/story/spectral-texture/paired';});
  await page.screenshot({path:`${outputDirectory}/texture-fit.png`,fullPage:true});
  await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:true,results,errors},null,2));
  console.log(JSON.stringify({passed:true,checks:results.length,errors}));
}catch(error){await writeFile(`${outputDirectory}/report.json`,JSON.stringify({passed:false,error:String(error),results,errors},null,2));throw error;}
finally{await browser.close(); await verifyBuildEvidence();}
