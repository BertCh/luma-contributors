// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/aviation-community-coarsening.json';
const dataDirectory = new URL('../public/data/us-airline-flows/', import.meta.url);
const referenceUrl = new URL(
  '../../../../dev-docs/research/showcase-overhaul-2026-10-25/aviation-network-reference.json',
  import.meta.url
);
const reference = JSON.parse(await readFile(referenceUrl, 'utf8'));

async function readTyped(name, Type) {
  const bytes = await readFile(new URL(name, dataDirectory));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

const [locations, origins, destinations, counts, airportFlights, airportRows] = await Promise.all([
  readTyped('position.bin', Float32Array),
  readTyped('origin.bin', Uint32Array),
  readTyped('destination.bin', Uint32Array),
  readTyped('count.bin', Uint32Array),
  readTyped('airportFlights.bin', Uint32Array),
  readFile(new URL('airports.csv', dataDirectory), 'utf8')
]);
const iataByAirport = airportRows
  .trim()
  .split('\n')
  .slice(1)
  .map(row => row.split(',')[1]);
const regionBounds = {
  west: [-125, 25, -105, 50],
  northeast: [-90, 35, -65, 50],
  south: [-105, 24, -75, 38]
};

function getInput(region) {
  const bounds = regionBounds[region];
  const selected = Array.from({length: airportFlights.length}, (_, airport) => airport)
    .filter(airport => {
      const longitude = locations[airport * 2];
      const latitude = locations[airport * 2 + 1];
      return longitude >= bounds[0] && longitude <= bounds[2] && latitude >= bounds[1] && latitude <= bounds[3];
    })
    .sort((first, second) => airportFlights[second] - airportFlights[first])
    .slice(0, 24);
  const localIndex = new Int32Array(airportFlights.length).fill(-1);
  selected.forEach((airport, index) => { localIndex[airport] = index; });
  const directedRows = [];
  const dense = new Float64Array(selected.length * selected.length);
  for (let row = 0; row < origins.length; row++) {
    const source = localIndex[origins[row]];
    const target = localIndex[destinations[row]];
    if (source < 0 || target < 0) continue;
    directedRows.push({source, target, weight: counts[row]});
    dense[source * selected.length + target] += counts[row];
  }
  return {
    airports: selected.map(airport => iataByAirport[airport]),
    directedRows,
    dense,
    nodeCount: selected.length
  };
}

function getExpected(region, model, memberGroups) {
  const input = getInput(region);
  const groupByIata = new Map();
  for (const [group, members] of memberGroups) {
    for (const member of members) {
      assert.ok(input.airports.includes(member), `${region}/${model}: unknown airport ${member}`);
      assert.equal(groupByIata.has(member), false, `${region}/${model}: duplicated ${member}`);
      groupByIata.set(member, group);
    }
  }
  assert.equal(groupByIata.size, input.nodeCount, `${region}/${model}: expected one community label per airport`);
  const localGroups = input.airports.map(iata => groupByIata.get(iata));
  let intraPairCount = 0;
  let interPairCount = 0;
  let intraWeight = 0;
  let interWeight = 0;
  if (model === 'directed') {
    for (const row of input.directedRows) {
      if (localGroups[row.source] === localGroups[row.target]) {
        intraPairCount++;
        intraWeight += row.weight;
      } else {
        interPairCount++;
        interWeight += row.weight;
      }
    }
  } else {
    for (let source = 0; source < input.nodeCount; source++) {
      for (let target = source; target < input.nodeCount; target++) {
        const forward = input.dense[source * input.nodeCount + target];
        const reverse = input.dense[target * input.nodeCount + source];
        const weight = source === target ? forward : forward + reverse;
        if (weight === 0) continue;
        if (localGroups[source] === localGroups[target]) {
          intraPairCount++;
          intraWeight += weight;
        } else {
          interPairCount++;
          interWeight += weight;
        }
      }
    }
  }
  return {input, inputWeight: input.directedRows.reduce((sum, row) => sum + row.weight, 0), groupCount: memberGroups.length, intraPairCount, interPairCount, intraWeight, interWeight};
}

function parseMembership(value) {
  const groups = value.split(' · ').filter(Boolean).map(entry => {
    const match = entry.match(/^(\d+): ([A-Z0-9-]+(?:,[A-Z0-9-]+)*)$/);
    assert.ok(match, `Invalid community membership text: ${entry}`);
    return [Number(match[1]), match[2].split(',')];
  });
  return groups;
}

function parseCoarsening(value) {
  const match = value.match(/^(\d+) groups · (\d+) inter-group pairs \/ (\d+) within-group pairs · weight ([\d,]+) within \+ ([\d,]+) between = ([\d,]+) \/ ([\d,]+) conserved$/);
  assert.ok(match, `Unexpected or incomplete coarsening readout: ${value}`);
  return {
    groupCount: Number(match[1]),
    interPairCount: Number(match[2]),
    intraPairCount: Number(match[3]),
    intraWeight: Number(match[4].replaceAll(',', '')),
    interWeight: Number(match[5].replaceAll(',', '')),
    totalWeight: Number(match[6].replaceAll(',', '')),
    inputWeight: Number(match[7].replaceAll(',', ''))
  };
}

// Validate the independent source reconstruction before acquiring a browser or GPU.
for (const region of Object.keys(regionBounds)) {
  const input = getInput(region);
  assert.equal(input.nodeCount, reference[region].airports, `${region}: source airport count`);
  assert.equal(input.directedRows.length, reference[region].arcs, `${region}: source directed pairs`);
  assert.equal(input.directedRows.reduce((sum, row) => sum + row.weight, 0), reference[region].scheduledFlightLegs, `${region}: source count weight`);
  for (const model of ['directed', 'symmetric']) {
    const expected = getExpected(region, model, [[0, input.airports]]);
    assert.equal(expected.intraWeight, reference[region].scheduledFlightLegs, `${region}/${model}: one-group conservation`);
    assert.equal(expected.interWeight, 0);
  }
}
if (process.argv.includes('--reference-only')) {
  console.log('Validated three regional source reconstructions and both graph-weight conventions without GPU.');
  process.exit(0);
}

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [];
const consoleErrors = [];
let page;
let passed = false;
let failureReason = null;
try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => consoleErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
  const url = new URL(baseUrl);
  url.hash = '#/story/aviation-network-lab';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null, 'Scene must compile before numerical checks');
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const region of Object.keys(regionBounds)) {
    for (const model of ['directed', 'symmetric']) {
      await page.locator('#all-option-region').selectOption(region);
      await page.locator('#all-option-measure').selectOption('communities');
      await page.locator('#all-option-communityModel').selectOption(model);
      await page.waitForFunction(({modelName, expectedArcs}) => {
        const rows = [...document.querySelectorAll('.readout')];
        return readReadout(rows, 'Louvain run').startsWith(modelName === 'directed' ? 'Directed Louvain' : 'Symmetric Louvain') &&
          readReadout(rows, 'GPU community membership').includes(':') &&
          readReadout(rows, 'GPU community-link aggregate').includes('conserved') &&
          readReadout(rows, 'Regional graph').includes(`${expectedArcs} directed`);
        function readReadout(readouts, label) {
          return readouts.find(row => row.querySelector('dt')?.textContent?.trim() === label)
            ?.querySelector('dd')?.textContent?.trim() ?? '';
        }
      }, {modelName: model, expectedArcs: reference[region].arcs}, {timeout: 90_000});
      const rows = await page.evaluate(() => [...document.querySelectorAll('.readout')].map(row => ({
        label: row.querySelector('dt')?.textContent?.trim() ?? '',
        value: row.querySelector('dd')?.textContent?.trim() ?? ''
      })));
      const values = Object.fromEntries(rows.map(row => [row.label, row.value]));
      const groups = parseMembership(values['GPU community membership']);
      const expected = getExpected(region, model, groups);
      const observed = parseCoarsening(values['GPU community-link aggregate']);
      assert.equal(observed.groupCount, expected.groupCount, `${region}/${model}: group count`);
      assert.equal(observed.intraPairCount, expected.intraPairCount, `${region}/${model}: within group pair count`);
      assert.equal(observed.interPairCount, expected.interPairCount, `${region}/${model}: between group pair count`);
      assert.equal(observed.intraWeight, expected.intraWeight, `${region}/${model}: within group weight`);
      assert.equal(observed.interWeight, expected.interWeight, `${region}/${model}: between group weight`);
      assert.equal(observed.totalWeight, expected.intraWeight + expected.interWeight, `${region}/${model}: coarsened total`);
      assert.equal(observed.inputWeight, Math.round(expected.inputWeight), `${region}/${model}: regional input weight`);
      assert.equal(observed.totalWeight, observed.inputWeight, `${region}/${model}: total weight conservation`);
      reports.push({region, model, observed, membership: values['GPU community membership']});
    }
  }
  await page.locator('#all-option-region').selectOption('west');
  await page.locator('#all-option-communityModel').selectOption('directed');
  for (const gamma of [0.5, 2]) {
    await page.locator('#all-option-gamma').evaluate((element, value) => {
      element.value = String(value);
      element.dispatchEvent(new Event('input', {bubbles: true}));
      element.dispatchEvent(new Event('change', {bubbles: true}));
    }, gamma);
    await page.waitForFunction(() => {
      const rows = [...document.querySelectorAll('.readout')];
      return readReadout(rows, 'GPU community-link aggregate').includes('conserved') && readReadout(rows, 'Regional graph').includes('338 directed') && readReadout(rows, 'Louvain run').startsWith('Directed Louvain');
      function readReadout(readouts, label) {
        return readouts.find(row => row.querySelector('dt')?.textContent?.trim() === label)
          ?.querySelector('dd')?.textContent?.trim() ?? '';
      }
    }, null, {timeout: 90_000});
    const rows = await page.evaluate(() => [...document.querySelectorAll('.readout')].map(row => ({
      label: row.querySelector('dt')?.textContent?.trim() ?? '',
      value: row.querySelector('dd')?.textContent?.trim() ?? ''
    })));
    const values = Object.fromEntries(rows.map(row => [row.label, row.value]));
    const groups = parseMembership(values['GPU community membership']);
    const expected = getExpected('west', 'directed', groups);
    const observed = parseCoarsening(values['GPU community-link aggregate']);
    assert.equal(observed.intraPairCount, expected.intraPairCount, `west/gamma${gamma}: intra pair count`);
    assert.equal(observed.interPairCount, expected.interPairCount, `west/gamma${gamma}: inter pair count`);
    assert.equal(observed.intraWeight, expected.intraWeight, `west/gamma${gamma}: intra weight`);
    assert.equal(observed.interWeight, expected.interWeight, `west/gamma${gamma}: inter weight`);
    reports.push({region: 'west', model: 'directed', gamma, observed});
  }
  assert.deepEqual(consoleErrors, [], 'Browser errors must not be hidden by numerical agreement');
  passed = true;
} catch (error) {
  failureReason = String(error);
  if (page) {
    await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true});
    reports.push({failure: failureReason, visibleText: await page.locator('body').innerText()});
  }
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, `${JSON.stringify({baseUrl, passed, status: passed ? 'passed' : 'failed', failureReason, reports, consoleErrors}, null, 2)}\n`);
}
console.log(`Verified ${reports.length} GPU community coarsening outputs: ${outputFile}`);
