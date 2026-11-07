// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/sampling-frames-browser-reference.json';
const rootUrl = new URL('../', import.meta.url);
const dataUrl = new URL('public/data/', rootUrl);
const reports = [];
const errors = [];
let browser;
let page;
let lastBody = '';
let lastReadouts = {};
let initialSeed;

const readJson = async url => JSON.parse(await readFile(url, 'utf8'));
const facilitiesManifest = await readJson(new URL('chicago-facilities/manifest.json', dataUrl));
const categories = new Uint8Array(await readFile(new URL('chicago-facilities/category.bin', dataUrl)));
const eligibleRows = Array.from(categories, (category, sourceRow) => ({category, sourceRow}))
  .filter(row => row.category === 1 || row.category === 3);
const eligibleByCategory = new Map([
  [1, eligibleRows.filter(row => row.category === 1)],
  [3, eligibleRows.filter(row => row.category === 3)]
]);
assert.equal(categories.length, facilitiesManifest.count, 'facility categories match manifest count');
assert.equal(eligibleRows.length, 174, 'binary-derived eligible frame size');
assert.equal(eligibleByCategory.get(1).length, 82, 'binary-derived library frame size');
assert.equal(eligibleByCategory.get(3).length, 92, 'binary-derived fire-station frame size');

// Match LocalMetricProjection: Web Mercator high-precision scales, one inverse refinement,
// then float32 storage for every projected boundary vertex before summing segment lengths.
const origin = [-87.6, 41.86];
const scales = getDistanceScales({longitude: origin[0], latitude: origin[1], highPrecision: true});
const metersPerDegree = [
  scales.unitsPerDegree[0] * scales.metersPerUnit[0],
  scales.unitsPerDegree[1] * scales.metersPerUnit[1]
];
const project = (longitude, latitude) => {
  let positionX = (longitude - origin[0]) * metersPerDegree[0];
  let positionY = (latitude - origin[1]) * metersPerDegree[1];
  const [estimatedLongitude, estimatedLatitude] = addMetersToLngLat(origin, [positionX, positionY]);
  positionX += (longitude - estimatedLongitude) * metersPerDegree[0];
  positionY += (latitude - estimatedLatitude) * metersPerDegree[1];
  return [positionX, positionY];
};
const parks = await readJson(new URL('chicago-parks/green-space.geojson', dataUrl));
const park = parks.features.find(feature => Number(feature.properties?.osmId) === 2914325);
assert.ok(park?.geometry?.type === 'Polygon', 'Northerly Island source OSM polygon exists');
const sourceOuter = park.geometry.coordinates[0];
const projectedOuter = new Float32Array(sourceOuter.length * 2);
for (let index = 0; index < sourceOuter.length; index++) {
  const [positionX, positionY] = project(sourceOuter[index][0], sourceOuter[index][1]);
  projectedOuter[index * 2] = positionX;
  projectedOuter[index * 2 + 1] = positionY;
}
let boundaryLength = 0;
for (let index = 2; index < projectedOuter.length; index += 2) {
  boundaryLength += Math.hypot(
    projectedOuter[index] - projectedOuter[index - 2],
    projectedOuter[index + 1] - projectedOuter[index - 1]
  );
}

const read = () => page.evaluate(() => {
  const values = Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? '']));
  return {...values, status: values['Design status']};
});
const bodyText = () => page.locator('body').innerText();
const isStartupFailure = text => /This analysis could not start|GPU graph could not compile|GPU graph build failed|GPU design error/.test(text);
const isPending = value => !value || /^(?:pending(?: replay)?|building GPU design…|—)$/i.test(value.trim());

const waitReady = async () => {
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready ||
    /This analysis could not start|GPU graph could not compile|GPU graph build failed/.test(document.body.innerText), null, {timeout: 30_000});
  const body = await bodyText();
  assert.ok(!isStartupFailure(body), `analysis startup failed: ${body}`);
  assert.ok(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), body);
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
};

const waitOutput = async (labels, timeout = 45_000) => {
  await page.waitForFunction(() => {
    if (/This analysis could not start|GPU graph could not compile|GPU graph build failed|GPU design error/.test(document.body.innerText)) return true;
    const row = [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === 'Design status');
    const value = row?.querySelector('dd')?.textContent ?? '';
    return value.trim() && !/^(?:pending(?: replay)?|building GPU design…|—)$/i.test(value.trim());
  }, null, {timeout});
  const body = await bodyText();
  assert.ok(!isStartupFailure(body), `GPU/design failure: ${body}`);
  const values = await read();
  lastBody = body;
  lastReadouts = values;
  assert.ok(values.status && !isPending(values.status), `Design status did not finish: ${values.status}`);
  void labels;
  return values;
};

const select = async (name, value) => {
  if (name === 'design') {
    const labels = {finite: 'Finite inventory', line: 'Continuous boundary', area: 'Continuous park area', 'iid-line': 'IID line baseline'};
    await page.locator('[aria-labelledby="all-option-design-label"]').getByRole('button', {name: labels[value], exact: true}).click();
  } else {
    await page.locator(`#all-option-${name}`).selectOption(String(value));
  }
};
const setToggle = async (name, value) => {
  const checkbox = page.locator(`#all-option-${name}`);
  if (await checkbox.isChecked() !== value) await checkbox.locator('..').click();
};
const setSlider = async (name, value) => page.locator(`#all-option-${name}`).evaluate((element, next) => {
  element.value = String(next);
  element.dispatchEvent(new Event('input', {bubbles: true}));
  element.dispatchEvent(new Event('change', {bubbles: true}));
}, value);
const ensureMode = async design => {
  await select('design', design);
  const values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
  assert.ok(!/withheld|error/i.test(values.status), values.status);
  return values;
};
const formatInteger = value => new Intl.NumberFormat('en-US', {maximumFractionDigits: 0}).format(value);
const near = (actual, expected, tolerance, message) => assert.ok(
  Math.abs(actual - expected) <= tolerance,
  `${message}: got ${actual}, expected ${expected} ± ${tolerance}`
);
const recordCase = async (name, action) => {
  const item = {name};
  try {
    item.readouts = await action();
    item.passed = true;
  } catch (error) {
    item.passed = false;
    item.error = String(error);
    item.readouts = await read().catch(() => lastReadouts);
    lastReadouts = item.readouts;
    lastBody = await bodyText().catch(() => lastBody);
    reports.push(item);
    errors.push(`${name}: ${String(error)}`);
    throw error;
  }
  reports.push(item);
  lastReadouts = item.readouts;
  return item.readouts;
};
const assertNoGpuErrors = () => assert.deepEqual(errors.filter(error => /^(?:pageerror|console):/.test(error)), []);

const verifyFinite = async (sampleCountValue, strata, unequal) => {
  const sampleCount = Number(sampleCountValue);
  await select('design', 'finite');
  await select('sampleCount', sampleCount);
  await select('strata', strata);
  await setToggle('unequal', unequal);
  const values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
  assert.match(values.status, new RegExp(`^valid · ${sampleCount} / 174 frame sites$`));
  assert.equal(Number(values['Valid sample rows']), sampleCount);
  const parsedIds = [...values['Selected IDs / provenance'].matchAll(/source row (\d+) · ID (\d+)/g)]
    .map(match => ({sourceRow: Number(match[1]), id: Number(match[2])}));
  assert.equal(parsedIds.length, sampleCount, 'every selected ID is shown');
  assert.equal(new Set(parsedIds.map(row => row.id)).size, sampleCount, 'selected IDs are unique');
  for (const row of parsedIds) {
    assert.equal(row.id, row.sourceRow + 1, 'displayed ID is source row + 1');
    assert.ok(eligibleRows.some(eligible => eligible.sourceRow === row.sourceRow), `eligible source row ${row.sourceRow}`);
  }
  if (strata === 'category') {
    const shownRows = parsedIds.map(row => eligibleRows.find(item => item.sourceRow === row.sourceRow));
    assert.equal(shownRows.filter(row => row.category === 1).length, sampleCount / 2);
    assert.equal(shownRows.filter(row => row.category === 3).length, sampleCount / 2);
  }

  const weightLines = values['Inclusion weights / units'].split('\n').slice(0, 8);
  assert.equal(weightLines.length, Math.min(8, sampleCount));
  for (const line of weightLines) {
    const match = line.match(/^ID (\d+): π=([\d.]+), 1\/π=([\d.]+)$/);
    assert.ok(match, line);
    const id = Number(match[1]);
    const entry = eligibleRows.find(row => row.sourceRow + 1 === id);
    assert.ok(entry, `displayed ID ${id} is source-derived`);
    const mass = unequal ? (entry.sourceRow % 2 === 0 ? 0.5 : 1.5) : 1;
    const allocation = strata === 'category' ? sampleCount / 2 : sampleCount;
    const stratumRows = strata === 'category' ? eligibleByCategory.get(entry.category) : eligibleRows;
    const ticketTotal = stratumRows.reduce((sum, row) => sum + (unequal ? (row.sourceRow % 2 === 0 ? 0.5 : 1.5) : 1), 0);
    const probability = allocation * mass / ticketTotal;
    near(Number(match[2]), probability, 0.00051, `formatted inclusion probability for source row ${entry.sourceRow}`);
    near(Number(match[3]), 1 / probability, 0.0051, `formatted design weight for source row ${entry.sourceRow}`);
  }

  const expectedTotal = parsedIds.reduce((sum, selected) => {
    const entry = eligibleRows.find(row => row.sourceRow === selected.sourceRow);
    assert.ok(entry, `selected source row ${selected.sourceRow} contributes to HT sum`);
    const allocation = strata === 'category' ? sampleCount / 2 : sampleCount;
    const stratumRows = strata === 'category' ? eligibleByCategory.get(entry.category) : eligibleRows;
    const mass = unequal ? (entry.sourceRow % 2 === 0 ? 0.5 : 1.5) : 1;
    const ticketTotal = stratumRows.reduce((subtotal, row) => subtotal + (unequal ? (row.sourceRow % 2 === 0 ? 0.5 : 1.5) : 1), 0);
    return sum + 1 / (allocation * mass / ticketTotal);
  }, 0);
  const estimateMatch = values['Weight sum / frame length'].match(/^Σ1\/π ([\d,]+) frame sites$/);
  assert.ok(estimateMatch, values['Weight sum / frame length']);
  assert.equal(Number(estimateMatch[1].replaceAll(',', '')), Number(formatInteger(expectedTotal)), 'HT estimate matches binary-derived frame total after display rounding');
  return values;
};

const getWaveRows = value => [...value.matchAll(/wave (\d+): (\d+) written \((\d+) random accepted, (\d+) legacy\); (\d+) required-so-far; flags (\d+)/g)]
  .map(match => ({wave: Number(match[1]), written: Number(match[2]), accepted: Number(match[3]), legacy: Number(match[4]), required: Number(match[5]), flags: Number(match[6])}));
const assertControlDisabled = async (name, disabled) => {
  assert.equal(await page.locator(`#all-option-${name}`).isDisabled(), disabled, `${name} disabled state`);
};
const verifyLine = async (spacing, waves, includeLegacy) => {
  await select('design', 'line');
  await select('spacing', spacing);
  await select('waves', waves);
  await select('capacity', '2048');
  await select('candidateBudget', '65536');
  await setToggle('includeLegacy', includeLegacy);
  const values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
  const waveRows = getWaveRows(values.status);
  assert.equal(waveRows.length, Number(waves), values.status);
  for (const row of waveRows) {
    assert.equal(row.flags, 0, values.status);
    assert.equal(row.written, row.required, values.status);
    assert.equal(row.written, row.accepted + row.legacy, values.status);
    const lower = Math.floor(boundaryLength / Number(spacing));
    const upper = Math.ceil(boundaryLength / Number(spacing));
    assert.ok(row.written >= lower && row.written <= upper, `phase-bounded line count ${row.written} in [${lower}, ${upper}]`);
    if (includeLegacy) assert.equal(row.legacy, 1, 'one fixed synthetic legacy site each wave');
    else assert.equal(row.legacy, 0);
  }
  const randomAccepted = waveRows.reduce((sum, row) => sum + row.accepted, 0);
  assert.equal(Number(values['Valid sample rows']), randomAccepted, 'selected count is random-accepted rows only');
  const lineWeights = values['Inclusion weights / units'].split('\n');
  assert.equal(lineWeights[0], `arc length ${spacing} m`);
  const shownWeights = lineWeights[1].split(', ').map(Number);
  assert.equal(shownWeights.length, Math.min(8, randomAccepted));
  for (const weight of shownWeights) near(weight, Number(spacing), 0.01, 'arc-length weight');
  near(Number(values['Weight sum / frame length'].split(' m')[0].replaceAll(',', '')), boundaryLength, 0.51, 'rounded mapped boundary length');
  return values;
};

const verifyArea = async includeLegacy => {
  await select('design', 'area');
  await select('spacing', '200');
  await select('waves', '1');
  await select('capacity', '2048');
  await select('candidateBudget', '65536');
  await setToggle('includeLegacy', includeLegacy);
  const values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
  const waveRows = getWaveRows(values.status);
  assert.equal(waveRows.length, 1, values.status);
  const row = waveRows[0];
  assert.equal(row.flags, 0, values.status);
  assert.equal(row.written, row.required, values.status);
  assert.equal(row.written, row.accepted + row.legacy, values.status);
  assert.ok(row.written >= 0 && row.accepted >= 0 && row.required >= 0, values.status);
  assert.equal(Number(values['Valid sample rows']), row.accepted, 'area selected count excludes zero-weight legacy row');
  if (includeLegacy) assert.equal(row.legacy, 1, 'one actual GPU triangulated-area legacy site');
  else assert.equal(row.legacy, 0);
  const areaWeights = values['Inclusion weights / units'].split('\n');
  assert.equal(areaWeights[0], 'area 40000 m²');
  for (const weight of areaWeights[1].split(', ').map(Number)) near(weight, 40000, 1, 'area expansion weight');
  assert.match(values['Weight sum / frame length'], /^[\d.]+ km² OSM area attribute$/);
  return {values, randomAccepted: row.accepted};
};

try {
  browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(`pageerror: ${error.message}`));
  page.on('console', message => {if (message.type() === 'error') errors.push(`console: ${message.text()}`);});
  const url = new URL(baseUrl);
  url.hash = '#/story/sampling-frames';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await waitReady();
  initialSeed = Number(await page.locator('#all-option-seed').inputValue());
  assert.ok(Number.isFinite(initialSeed), 'initial seed is read from the rendered control');
  const defaultValues = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
  assert.match(defaultValues.status, /^valid · 10 \/ 174 frame sites$/);
  assert.equal(Number(defaultValues['Valid sample rows']), 10);

  for (const sampleCount of ['10', '20']) for (const strata of ['category', 'one']) for (const unequal of [false, true]) {
    await recordCase(`finite-${sampleCount}-${strata}-${unequal ? 'unequal' : 'equal'}`, () => verifyFinite(sampleCount, strata, unequal));
  }

  // Same design key must replay IDs, and replicate 12 must be observably a different draw.
  await recordCase('finite-seed-replay-11-12-11', async () => {
    await select('design', 'finite');
    await select('sampleCount', '20');
    await select('strata', 'category');
    await setToggle('unequal', false);
    await setSlider('seed', 11);
    await setSlider('replicate', 0);
    let values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    const first = values['Selected IDs / provenance'];
    await setSlider('seed', 12);
    values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    const alternate = values['Selected IDs / provenance'];
    assert.notEqual(alternate, first, 'seed change changes finite sample sequence');
    await setSlider('seed', 11);
    values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    assert.equal(values['Selected IDs / provenance'], first, 'restoring seed replays same finite ID sequence');
    return values;
  });

  await recordCase('finite-rapid-replicates-final-readout-current', async () => {
    await setSlider('seed', 11);
    await setSlider('replicate', 2);
    let values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    const expectedFinal = values['Selected IDs / provenance'];
    await setSlider('replicate', 0);
    await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    await setSlider('replicate', 1);
    await setSlider('replicate', 2);
    values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    assert.equal(values['Selected IDs / provenance'], expectedFinal, 'rapid 0→1→2 updates leave replicate 2 as the current readout');
    return values;
  });

  for (const spacing of ['100', '200', '400']) {
    for (const waves of spacing === '200' ? ['1', '2'] : ['1']) {
      const noLegacy = await recordCase(`line-${spacing}-waves-${waves}-no-legacy`, () => verifyLine(spacing, waves, false));
      const noLegacyRows = getWaveRows(noLegacy.status);
      const legacy = await recordCase(`line-${spacing}-waves-${waves}-legacy`, async () => {
        const values = await verifyLine(spacing, waves, true);
        const legacyRows = getWaveRows(values.status);
        assert.equal(legacyRows.length, noLegacyRows.length);
        legacyRows.forEach((row, index) => {
          assert.equal(row.accepted, noLegacyRows[index].accepted - 1, `legacy displaces one random point for wave ${index}`);
          assert.equal(row.written, noLegacyRows[index].written, `legacy keeps written count unchanged for wave ${index}`);
          assert.equal(row.legacy, 1);
        });
        return values;
      });
      const legacyRows = getWaveRows(legacy.status);
      assert.equal(legacyRows.length, noLegacyRows.length, 'paired case reported matching waves');
    }
  }

  const areaNoLegacy = await recordCase('area-200-no-legacy', async () => (await verifyArea(false)).values);
  const areaNoLegacyRows = getWaveRows(areaNoLegacy.status);
  const areaLegacy = await recordCase('area-200-legacy', async () => {
    const values = (await verifyArea(true)).values;
    const rows = getWaveRows(values.status);
    assert.ok(rows[0].accepted <= areaNoLegacyRows[0].accepted, 'fixed legacy site cannot increase random area lattice count');
    return values;
  });
  const areaLegacyRows = getWaveRows(areaLegacy.status);
  assert.equal(areaLegacyRows.length, areaNoLegacyRows.length, 'paired area cases report matching wave counts');

  await recordCase('line-capacity-flag-2-withheld', async () => {
    await select('design', 'line');
    await select('spacing', '200');
    await select('waves', '1');
    await select('capacity', '1');
    await select('candidateBudget', '65536');
    await setToggle('includeLegacy', false);
    const values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    const rows = getWaveRows(values.status);
    assert.equal(rows.length, 1, values.status);
    assert.ok((rows[0].flags & 2) !== 0, `capacity flag bit 2: ${values.status}`);
    assert.equal(values['Valid sample rows'], '—', 'invalid capacity withholds selected count');
    assert.match(values['Selected IDs / provenance'], /withheld/);
    return values;
  });

  await recordCase('line-candidate-budget-flag-4-withheld', async () => {
    await select('design', 'line');
    await select('spacing', '200');
    await select('waves', '1');
    await select('capacity', '256');
    await select('candidateBudget', '1');
    await setToggle('includeLegacy', false);
    const values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    const rows = getWaveRows(values.status);
    assert.equal(rows.length, 1, values.status);
    assert.ok((rows[0].flags & 4) !== 0, `candidate budget flag bit 4: ${values.status}`);
    assert.equal(values['Valid sample rows'], '—', 'incomplete candidate output withholds selected count');
    assert.match(values['Selected IDs / provenance'], /withheld/);
    return values;
  });

  await recordCase('line-valid-after-restoring-limits', () => verifyLine('200', '1', false));

  await recordCase('iid-prefix-8', async () => {
    await select('design', 'iid-line');
    await select('iidCount', '8');
    await setSlider('seed', 47021);
    await setSlider('replicate', 0);
    const values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    assert.equal(Number(values['Valid sample rows']), 8);
    assert.match(values['Selected IDs / provenance'], /^line ids 0(?:, 0){7}$/);
    assert.match(values['Inclusion weights / units'], /^fractions (?:0\.\d{3})(?:, 0\.\d{3}){7}$/);
    return values;
  });
  const iid8 = lastReadouts;
  await recordCase('iid-prefix-16', async () => {
    await select('design', 'iid-line');
    await select('iidCount', '16');
    const values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    assert.equal(Number(values['Valid sample rows']), 16);
    assert.equal(values['Selected IDs / provenance'].match(/\b0\b/g)?.length, 16);
    assert.equal(values['Inclusion weights / units'].split(', ').length, 8, 'displayed fractions are a prefix only');
    assert.equal(values['Inclusion weights / units'], iid8['Inclusion weights / units'], 'same-seed first-eight fractions preserved');
    return values;
  });
  const iid16 = lastReadouts;
  await recordCase('iid-prefix-32', async () => {
    await select('design', 'iid-line');
    await select('iidCount', '32');
    const values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    assert.equal(Number(values['Valid sample rows']), 32);
    assert.equal(values['Selected IDs / provenance'].match(/\b0\b/g)?.length, 32);
    assert.equal(values['Inclusion weights / units'], iid16['Inclusion weights / units'], 'same-seed first-eight fractions preserved at 32');
    return values;
  });
  await recordCase('iid-seed-change-and-restore', async () => {
    await setSlider('seed', 47022);
    let values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    const changed = values['Inclusion weights / units'];
    assert.notEqual(changed, iid16['Inclusion weights / units'], 'seed change alters IID fraction stream');
    await setSlider('seed', 47021);
    values = await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
    assert.equal(values['Inclusion weights / units'], iid16['Inclusion weights / units'], 'restoring seed replays IID fraction prefix');
    return values;
  });

  await recordCase('irrelevant-controls-disabled', async () => {
    await ensureMode('finite');
    for (const name of ['iidCount', 'spacing', 'waves', 'capacity', 'candidateBudget', 'includeLegacy']) await assertControlDisabled(name, true);
    await ensureMode('line');
    for (const name of ['sampleCount', 'strata', 'unequal', 'iidCount']) await assertControlDisabled(name, true);
    await ensureMode('area');
    for (const name of ['sampleCount', 'strata', 'unequal', 'iidCount']) await assertControlDisabled(name, true);
    await ensureMode('iid-line');
    for (const name of ['sampleCount', 'strata', 'unequal', 'spacing', 'waves', 'capacity', 'candidateBudget', 'includeLegacy']) await assertControlDisabled(name, true);
    return await waitOutput(['status', 'selected', 'ids', 'weights', 'estimate']);
  });
  assertNoGpuErrors();
} catch (error) {
  if (!reports.some(item => !item.passed)) errors.push(`harness: ${String(error)}`);
} finally {
  lastBody = page ? await bodyText().catch(() => lastBody) : lastBody;
  if (page) lastReadouts = await read().catch(() => lastReadouts);
  if (browser) await browser.close().catch(error => errors.push(`browser close: ${String(error)}`));
  await writeFile(outputFile, JSON.stringify({
    baseUrl,
    sourceReference: {
      initialSeed,
      facilityCount: facilitiesManifest.count,
      eligibleCount: eligibleRows.length,
      categoryCounts: Object.fromEntries([...eligibleByCategory].map(([category, rows]) => [category, rows.length])),
      sourceOsmId: Number(park.properties.osmId),
      boundaryLengthMeters: boundaryLength,
      areaExpectedCountContract: 'Actual GPU triangulated polygon; point count is reported and contract-checked, not independently predicted without coordinates.'
    },
    reports,
    errors,
    lastBody,
    lastReadouts
  }, null, 2) + '\n');
}

assert.deepEqual(errors, [], `Sampling frames acceptance failed; report written to ${outputFile}`);
assert.ok(reports.every(item => item.passed), `Sampling frames cases failed; report written to ${outputFile}`);
assert.equal(reports.length, 28, 'bounded meaningful acceptance case count');
console.log(`Accepted ${reports.length} sampling frames cases; report: ${outputFile}`);
