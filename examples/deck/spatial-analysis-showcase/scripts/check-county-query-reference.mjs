// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {dirname} from 'node:path';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/county-query-reference.json';
await mkdir(dirname(outputFile), {recursive: true});
const evidenceUrl = new URL('build-evidence.json', baseUrl);
const initialEvidenceResponse = await fetch(evidenceUrl, {cache: 'no-store'});
assert.ok(initialEvidenceResponse.ok, 'Production build evidence must be available');
const initialEvidence = await initialEvidenceResponse.json();
assert.match(initialEvidence.bundleSha256, /^[a-f0-9]{64}$/);
await writeFile(outputFile.replace(/\.json$/, '-build-evidence.json'), JSON.stringify(initialEvidence, null, 2));
const reference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/county-query-reference.json', import.meta.url), 'utf8'));
const expressionReference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/county-expression-reference.json', import.meta.url), 'utf8'));
const nanReference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/county-nan-sort-reference.json', import.meta.url), 'utf8'));
const countyNames = JSON.parse(await readFile(new URL('../public/data/us-counties/names.json', import.meta.url), 'utf8'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless:true,backend:'webgpu-core'}));
const page = await browser.newPage({viewport:{width:1440,height:960}});
const errors = [];
const reports = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
const readouts = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent??'',row.querySelector('dd')?.textContent??''])));
async function option(id, name) {
  await page.locator(`[role="radiogroup"][aria-labelledby="all-option-${id}-label"]`).getByRole('radio',{name,exact:true}).click();
}
async function waitReadout(label, text) {
  await page.waitForFunction(({label,text}) => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent===label && row.querySelector('dd')?.textContent?.includes(text)),{label,text},{timeout:90_000});
}
try {
  const url = new URL(baseUrl); url.hash = '#/story/county-query-semantics';
  await page.goto(url.toString(),{waitUntil:'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || document.body.textContent?.includes('This analysis could not start:'),null,{timeout:60_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), true, await page.locator('body').innerText());
  await page.getByRole('tab',{name:'All controls',exact:false}).click();
  await waitReadout('Finite 2024 observations','2,859');
  let observed = await readouts();
  assert.match(observed['Polygon/FIPS rows'],/3,109/);
  assert.match(observed['Nulls / lookup misses'],/250 explicit nulls; 250 lookup misses/);
  assert.match(observed['GPU validity predicates'],/isValid: 2016 3,100, 2020 3,100, 2024 2,859; 2024 isNull: 250/);
  assert.match(observed['2020→2024 share change'],/2,859 counties/);
  assert.match(observed['2020→2024 share change'],/-1\.68 percentage points/);
  assert.match(observed['Capacity / uniqueness'],/all outputs fit; unique keys/);
  for (const [state, abbreviation] of [[2,'AK'],[6,'CA'],[12,'FL'],[15,'HI'],[36,'NY'],[48,'TX']]) {
    const expected = reference.stateGroups2024[state];
    assert.ok(observed['State-FIPS group summary'].includes(`${abbreviation} n=${expected.count} mean=${expected.count ? `${(expected.mean * 100).toFixed(1)}%` : 'unavailable'}`));
  }
  reports.push({scenario:'source-and-derived',observed});
  await waitReadout('2024 share expressions', '2,859 valid counties');
  observed = await readouts();
  assert.ok(observed['2024 share expressions'].includes(`mean Republican complement ${(expressionReference.republicanComplement.mean * 100).toFixed(2)}%`));
  assert.ok(observed['2024 share expressions'].includes(`mean normalized margin ${expressionReference.normalizedDemocraticMargin.mean.toFixed(4)}`));
  assert.equal(observed['Three-valued NOT predicates'], `NOT(share > 0.5): ${expressionReference.notDemocraticMajority.toLocaleString('en-US')}; isNull OR NOT: ${expressionReference.missingOrNotDemocraticMajority.toLocaleString('en-US')} including unknowns`);
  assert.equal(observed['Exact cutoff and null boundary'], `Exactly 0.5: ${expressionReference.tiedTwoPartyShare} ties; notEqual(0.5): ${expressionReference.notEqualHalf.toLocaleString('en-US')} finite non-ties (nulls stay unknown)`);
  reports.push({scenario:'nullable-share-expressions',observed});
  for (const [join,label,count] of [['inner','Inner',2859],['left','Left',3109],['semi','Semi',2859],['anti','Anti',250]]) {
    await option('join',label);
    await waitReadout('Selected join result',`${join} ${count.toLocaleString('en-US')}`);
    observed = await readouts();
    assert.match(observed['Capacity / uniqueness'],/all outputs fit; unique keys/);
    reports.push({scenario:`join-${join}`,observed});
  }
  await option('nulls','Last');
  for (const [direction,label] of [['ascending','Low first'],['descending','High first']]) {
    await option('sortDirection',label);
    const expected = reference.years['2024'].ranks[direction];
    await waitReadout('Global full-sort leading rows',expected[0].name);
    observed = await readouts();
    const rows = observed['Global full-sort leading rows'].split('\n');
    assert.equal(rows.length,8);
    rows.forEach((row,index) => {
      assert.ok(row.includes(expected[index].name),`Rank${index+1}: ${row}; expected ${expected[index].name}`);
      assert.ok(row.includes(`${(expected[index].value*100).toFixed(1)}%`));
    });
    for (const expectedBatch of reference.stateBatchHeads2024[direction]) {
      assert.ok(observed['First county from state-batch sorts'].includes(expectedBatch.name));
    }
    reports.push({scenario:`full-sort-${direction}-nulls-last`,observed});
  }
  await option('nulls','First');
  await waitReadout('Global full-sort leading rows','null');
  observed = await readouts();
  assert.equal(observed['Global full-sort leading rows'].split('\n').filter(row=>row.includes('null')).length,8);
  reports.push({scenario:'full-sort-nulls-first',observed});
  const labelRow = ({row}) => `${countyNames.name[row]}, ${countyNames.state[row]}`;
  for (const variant of nanReference.variants) {
    const diagnostic = variant.representation === 'valid-nan';
    await option('nanMode', diagnostic ? 'Valid NaN diagnostic' : 'Unavailable / null');
    await option('sortDirection', variant.direction === 'ascending' ? 'Low first' : 'High first');
    await option('nulls', variant.placement === 'first' ? 'First' : 'Last');
    const expectedGlobal = variant.globalFirstEight.map(labelRow);
    const expectedTail = diagnostic ? variant.globalLastEight.map(labelRow) : [];
    const expectedBatches = variant.stateBatchFirstRows.map(labelRow);
    const expectedValidity = diagnostic ? '3,109 valid / 0 null' : '2,859 valid / 250 null';
    await page.waitForFunction(({expectedGlobal, expectedTail, expectedBatches, expectedValidity}) => {
      const values = Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? '']));
      return values['Active validity representation']?.includes(expectedValidity)
        && [...expectedGlobal, ...expectedTail].every(label => values['Global full-sort leading rows']?.includes(label))
        && expectedBatches.every(label => values['First county from state-batch sorts']?.includes(label));
    }, {expectedGlobal, expectedTail, expectedBatches, expectedValidity}, {timeout:15_000});
    observed = await readouts();
    const globalLines = observed['Global full-sort leading rows'].split('\n');
    const head = diagnostic ? globalLines.slice(1, 9) : globalLines;
    assert.equal(head.length, 8);
    head.forEach((line, index) => assert.ok(line.startsWith(expectedGlobal[index]), `Global head ${index}: ${line}`));
    if (diagnostic) {
      const tail = globalLines.slice(10);
      assert.equal(tail.length, 8);
      tail.forEach((line, index) => assert.ok(line.startsWith(expectedTail[index]), `Global tail ${index}: ${line}`));
      assert.ok(observed['Active validity representation'].includes('unavailable observations'));
    }
    const batchLines = observed['First county from state-batch sorts'].split('\n');
    assert.equal(batchLines.length, expectedBatches.length);
    batchLines.forEach((line, index) => assert.ok(line.includes(expectedBatches[index]), `State head ${index}: ${line}`));
    assert.match(observed['Finite 2024 observations'], /2,859/);
    reports.push({scenario:`representation-${variant.representation}-${variant.direction}-${variant.placement}`,observed});
  }
  await option('nanMode', 'Unavailable / null');
  await waitReadout('Active validity representation', '2,859 valid / 250 null');
  assert.deepEqual(errors,[]);
  await page.screenshot({path:outputFile.replace(/\.json$/,'-verified.png')});
} catch(error) {
  reports.push({failure:String(error),visibleText:await page.locator('body').innerText()});
  await page.screenshot({path:outputFile.replace(/\.json$/,'-failure.png'),fullPage:true});
  throw error;
} finally {
  await browser.close();
  const finalEvidenceResponse = await fetch(evidenceUrl, {cache: 'no-store'});
  assert.ok(finalEvidenceResponse.ok, 'Production build evidence must remain available');
  const finalEvidence = await finalEvidenceResponse.json();
  const bundleUnchanged = finalEvidence.bundleSha256 === initialEvidence.bundleSha256;
  await writeFile(outputFile,JSON.stringify({baseUrl,reports,errors,bundleSha256:initialEvidence.bundleSha256,finalBundleSha256:finalEvidence.bundleSha256,bundleUnchanged,passed:bundleUnchanged && reports.length===17 && errors.length===0 && !reports.some(report=>report.failure)},null,2)+'\n');
  assert.ok(bundleUnchanged, 'The production bundle changed during numerical verification');
}
console.log(`Verified ${reports.length} county-query scenarios: ${outputFile}`);
