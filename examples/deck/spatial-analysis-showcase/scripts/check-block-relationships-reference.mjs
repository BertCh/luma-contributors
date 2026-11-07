// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/block-relationships-browser-reference.json';
const referenceFile = '/tmp/block-relationships-independent-reference.json';
const referenceScript = new URL('./block-relationships-reference.mjs', import.meta.url);
execFileSync(
  process.execPath,
  [referenceScript.pathname, '/tmp/ahk-showcase-venv/bin/python', referenceFile],
  {stdio: 'inherit'}
);
const reference = JSON.parse(await readFile(referenceFile, 'utf8'));
const reports = [];
const errors = [];
const capabilities = {
  independentCpuComparisons: [
    'bounds intersects/within/contains at half-extents 8, 12, 20 and 48 m',
    'all three unordered shared-boundary lengths, connected components and positive-length neighbor counts',
    'visibility degree, reachability, mean-hop distance, hop closeness, choice, normalized choice and Hillier-Hanson integration at 25, 55 and 100 m',
    'visibility edge-list capacity 8 versus 120 while dense metrics remain unchanged',
    'tessellation sample count/completeness and required displayed total cell area against the enclosure'
  ],
  contractChecks: [
    'remaining visibility metrics expose only finite or explicitly undefined values',
    '16-site tessellation is withheld when incomplete',
    'zero-inset duplicate-site outcome reports failure/incompleteness honestly',
    'recovery and view/cache/query changes preserve completed readouts'
  ],
  unavailableDiagnostics: []
};
const smoke = process.argv.includes('--smoke');
const expectedCaseCount = smoke ? 3 : 12 + 1 + 21 + 6 + 1 + 3 + 1 + 1 + 1 + 1;
let browser;
let page;
let lastBody = '';
let lastReadouts = {};
let areaReadoutAvailable = false;

const readVisibleReadouts = () =>
  page.evaluate(() => {
    const readouts = {};
    for (const element of document.querySelectorAll('[data-readout-value]')) {
      const readoutId = element.getAttribute('data-readout-value');
      if (!readoutId) continue;
      (readouts[readoutId] ??= []).push(element.innerText ?? element.textContent ?? '');
    }
    const chartTitles = {};
    for (const readoutId of ['blockAdjacencyChart', 'blockTessellationChart', 'blockVisibilityChart']) {
      chartTitles[readoutId] = [...document.querySelectorAll(`[data-readout-value="${readoutId}"] .chart-bar > title`)]
        .map(element => element.textContent ?? '');
    }
    return {readouts, chartTitles};
  });

const getOptionValues = async optionIds =>
  page.evaluate(optionIds => Object.fromEntries(optionIds.map(optionId => {
    const element = document.querySelector(`#all-option-${optionId}`);
    return [optionId, element?.value ?? null];
  })), optionIds);

const waitForOutput = async (optionValues, predicate) => {
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    const selectedOptions = await getOptionValues(Object.keys(optionValues));
    const snapshot = await readVisibleReadouts();
    const body = await page.locator('body').innerText();
    if (/This analysis could not start|graph failed:|GPUBufferSurface output could not be packed|Per-building buffer preparation failed/i.test(body)) throw new Error(body);
    if (JSON.stringify(selectedOptions) === JSON.stringify(optionValues) && predicate(snapshot.readouts)) {
      // The selected option and result predicate must match together; two frames let async readout
      // rendering finish before the terminal snapshot is captured.
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      lastReadouts = await readVisibleReadouts();
      return lastReadouts;
    }
    await page.waitForTimeout(100);
  }
  throw new Error(`Timed out waiting for selected options ${JSON.stringify(optionValues)} and matching readout metadata`);
};

const runCase = async (name, action) => {
  const item = {name};
  try {
    item.result = await action();
    item.readouts = lastReadouts;
    item.passed = true;
  } catch (error) {
    item.passed = false;
    item.error = String(error);
    item.readouts = await readVisibleReadouts().catch(() => lastReadouts);
    lastReadouts = item.readouts;
  }
  reports.push(item);
  if (!item.passed) throw new Error(`${name}: ${item.error}`);
};

const selectOption = async (optionId, value) => {
  const previousValue = await page.locator(`#all-option-${optionId}`).inputValue();
  const previousRebuildCount = await page.evaluate(() => globalThis.spatialAnalysisShowcase?.rebuildCount ?? 0);
  await page.locator(`#all-option-${optionId}`).selectOption(String(value));
  if (previousValue !== String(value) && ['boundsRelation', 'inset', 'visibilityDistance', 'visibilityEdgeCapacity'].includes(optionId)) {
    await page.waitForFunction(
      previousRebuildCount => (globalThis.spatialAnalysisShowcase?.rebuildCount ?? 0) > previousRebuildCount,
      previousRebuildCount,
      {timeout: 120_000}
    );
  }
};

const setRangeOption = async (optionId, value) => {
  await page.locator(`#all-option-${optionId}`).evaluate((element, nextValue) => {
    element.value = String(nextValue);
    element.dispatchEvent(new Event('input', {bubbles: true}));
    element.dispatchEvent(new Event('change', {bubbles: true}));
  }, value);
};

const navigateToStep = async stepId => {
  const url = new URL(baseUrl);
  url.hash = `#/story/block-relationships/${stepId}`;
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => globalThis.spatialAnalysisShowcase?.ready || document.body.innerText.includes('This analysis could not start'),
    null,
    {timeout: 120_000}
  );
  assert.ok(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), await page.locator('body').innerText());
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
};

const waitForReadout = async (optionValues, readoutId, predicate) => {
  const snapshot = await waitForOutput(optionValues, readouts =>
    (readouts[readoutId] ?? []).some(value => predicate(value))
  );
  const value = (snapshot.readouts[readoutId] ?? []).find(predicate) ?? '';
  assert.ok(value, `Missing ${readoutId} readout`);
  return value;
};

const parseProbeValues = (summary, expectedValues) => {
  const values = [...summary.matchAll(/P(\d+) (undefined|[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)/gi)];
  assert.equal(values.length, 16, `Expected all 16 probe values: ${summary}`);
  return values.map((match, index) => {
    assert.equal(Number(match[1]), index + 1, `Probe ordering in ${summary}`);
    const expectedValue = expectedValues[index];
    if (expectedValue === null) {
      assert.equal(match[2], 'undefined', `Undefined contract for probe ${index + 1}: ${summary}`);
      return null;
    }
    const actualValue = Number(match[2]);
    assert.ok(Number.isFinite(actualValue), `Finite probe ${index + 1}: ${summary}`);
    assert.ok(Math.abs(actualValue - expectedValue) <= 0.002, `Probe ${index + 1}: expected ${expectedValue}, got ${actualValue}`);
    return actualValue;
  });
};

const visibilityGraphs = new Map(reference.visibility.maximumDistanceGraphs.map(graph => [graph.maximumDistance, graph]));
const visibilityMetricFields = {
  degree: 'degree',
  reachability: 'reachableVertices',
  'mean-hop-distance': 'meanHopDistance',
  'hop-closeness': 'closeness',
  choice: 'choice',
  'normalized-choice': 'normalizedChoice',
  'hillier-integration': 'hillierHansonIntegration'
};
const visibilityMetricLabels = {
  degree: 'visible neighbors',
  reachability: 'reachable probes',
  'mean-hop-distance': 'mean shortest-path distance (hops)',
  'hop-closeness': 'reachable-hop closeness',
  choice: 'betweenness choice',
  'normalized-choice': 'normalized choice',
  'hillier-integration': 'Hillier-Hanson integration'
};
const boundsCases = reference.boundsFilter.relations;
const expectedSamples = new Map(
  reference.boundarySampling.map(row => [`${row.geometry}-${row.spacing}-${row.capacity}`, row])
);

const assertVisibilitySummary = (summary, maximumDistance, metric, selectedEdgeCapacity) => {
  const graph = visibilityGraphs.get(maximumDistance);
  assert.ok(graph, `Independent visibility graph for ${maximumDistance} m`);
  assert.ok(summary.includes(`Selected ${visibilityMetricLabels[metric]} by probe:`), summary);
  const edgeCount = graph.edges.length;
  const expectedWritten = Math.min(edgeCount, selectedEdgeCapacity);
  const edgeCounts = summary.match(/(\d+) \/ (\d+) edges in bounded drawn list \(overflow (\d+)\)/);
  assert.ok(edgeCounts, summary);
  assert.equal(Number(edgeCounts[1]), expectedWritten, summary);
  assert.equal(Number(edgeCounts[2]), edgeCount, summary);
  assert.equal(Number(edgeCounts[3]), Number(edgeCount > selectedEdgeCapacity), summary);
  assert.match(summary, /invalid\/blocked classifications 0\./);
  const field = visibilityMetricFields[metric];
  const expectedValues = graph.graphMetrics[field];
  assert.equal(expectedValues.length, 16);
  return parseProbeValues(summary, expectedValues);
};

const parseChartValues = titles => {
  const rowsByLabel = new Map();
  for (const title of titles) {
    const match = title.match(/^([^:]+):\s*([\d,.]+)(K|M|B)?$/);
    if (!match) continue;
    const multiplier = match[3] === 'K' ? 1e3 : match[3] === 'M' ? 1e6 : match[3] === 'B' ? 1e9 : 1;
    rowsByLabel.set(match[1].trim(), Number(match[2].replaceAll(',', '')) * multiplier);
  }
  return [...rowsByLabel].map(([label, value]) => ({label, value}));
};

try {
  browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(`pageerror: ${error.message}`));
  page.on('console', message => {if (message.type() === 'error') errors.push(`console: ${message.text()}`);});

  await navigateToStep('bounds-candidates');
  await selectOption('view', 'bounds');
  const selectedBoundsExtents = smoke ? [8] : [8, 12, 20, 48];
  for (const relation of (smoke ? ['intersects', 'within', 'intersects'] : ['intersects', 'within', 'contains'])) {
    for (const halfExtent of selectedBoundsExtents) {
      await runCase(`bounds-${relation}-${halfExtent}m`, async () => {
        await selectOption('boundsRelation', relation);
        await setRangeOption('queryHalfExtent', halfExtent);
        const expectedMatches = boundsCases[relation].find(row => row.halfExtent === halfExtent).selectedSourceIds;
        const summary = await waitForReadout(
          {view: 'bounds', boundsRelation: relation, queryHalfExtent: String(halfExtent)},
          'blockBounds',
          value => value.startsWith(`GPUBoundsFilter ${relation}: ${expectedMatches.length} / 3 footprints in query;`)
        );
        assert.match(summary, /overflow 0\./);
        return {summary, expectedSourceIds: expectedMatches};
      });
    }
  }

  if (!smoke) {
  await navigateToStep('mapped-adjacency');
  await selectOption('view', 'adjacency');
  await runCase('three-pair-adjacency-topology', async () => {
    const summary = await waitForReadout(
      {view: 'adjacency'},
      'blockAdjacency',
      value => value.includes('GPU adjacency recipe complete;') && value.includes('Per-building morphology:')
    );
    assert.match(summary, /GPU adjacency recipe complete;/);
    const componentMatch = summary.match(/connected-component labels ([\d, ]+)\./);
    assert.ok(componentMatch, summary);
    const componentLabels = componentMatch[1].split(',').map(Number);
    assert.equal(componentLabels.length, 3, summary);
    assert.ok(componentLabels.every(label => label === componentLabels[0]), `One CPU-reference connected component: ${summary}`);

    const expectedPairs = reference.sharedBoundaries;
    for (const pair of expectedPairs) {
      const pairText = `${pair.leftSourceId} ↔ ${pair.rightSourceId}: `;
      const pairStart = summary.indexOf(pairText);
      assert.notEqual(pairStart, -1, `Shared pair ${pairText} in ${summary}`);
      const pairLine = summary.slice(pairStart).match(/^([^;]+)/)?.[1] ?? '';
      const length = Number(pairLine.match(/: ([\d.]+) m mapped shared boundary/)?.[1]);
      assert.ok(Number.isFinite(length), pairLine);
      assert.ok(Math.abs(length - pair.sharedBoundaryLength) <= 0.015, `${pairText}${length} vs CPU ${pair.sharedBoundaryLength}`);
    }
    const expectedNeighbors = new Map([[342227, 2], [815550, 1], [815551, 1]]);
    for (const [sourceId, expectedCount] of expectedNeighbors) {
      const morphology = summary.match(new RegExp(`${sourceId}: (\\d+) positive-length mapped neighbors`));
      assert.ok(morphology, summary);
      assert.equal(Number(morphology[1]), expectedCount, summary);
    }
    const chartValues = parseChartValues(lastReadouts.chartTitles.blockAdjacencyChart ?? []);
    assert.equal(chartValues.length, 3, `Adjacency chart diagnostics: ${JSON.stringify(lastReadouts.chartTitles.blockAdjacencyChart)}`);
    for (const [index, pair] of expectedPairs.entries()) {
      assert.ok(Math.abs(chartValues[index].value - pair.sharedBoundaryLength) <= 0.051, JSON.stringify(chartValues[index]));
    }
    return {summary, componentLabels, chartValues};
  });

  const knownVisibilityMetrics = Object.keys(visibilityMetricFields);
  for (const maximumDistance of [25, 55, 100]) {
    for (const metric of knownVisibilityMetrics) {
      await runCase(`visibility-${maximumDistance}m-${metric}`, async () => {
        await navigateToStep('synthetic-sightlines');
        await selectOption('view', 'visibility');
        await selectOption('visibilityDistance', maximumDistance);
        await selectOption('visibilityEdgeCapacity', '120');
        await selectOption('visibilityMetric', metric);
        const summary = await waitForReadout(
          {view: 'visibility', visibilityDistance: String(maximumDistance), visibilityEdgeCapacity: '120', visibilityMetric: metric},
          'blockVisibility',
          value => value.includes(`Selected ${visibilityMetricLabels[metric]} by probe:`)
        );
        const values = assertVisibilitySummary(summary, maximumDistance, metric, 120);
        return {summary, values};
      });
    }
  }

  for (const metric of ['relative-asymmetry', 'real-relative-asymmetry', 'pyramid', 'teklenburg', 'penn', 'maximum-depth']) {
    await runCase(`visibility-55m-${metric}-finite-or-undefined-contract`, async () => {
      await navigateToStep('synthetic-sightlines');
      await selectOption('view', 'visibility');
      await selectOption('visibilityDistance', '55');
      await selectOption('visibilityEdgeCapacity', '120');
      await selectOption('visibilityMetric', metric);
      const summary = await waitForReadout(
        {view: 'visibility', visibilityDistance: '55', visibilityEdgeCapacity: '120', visibilityMetric: metric},
        'blockVisibility',
        value => value.includes('Selected ') && value.includes(' by probe:')
      );
      const probeValues = [...summary.matchAll(/P(\d+) (undefined|[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)/gi)];
      assert.equal(probeValues.length, 16, summary);
      for (const [index, match] of probeValues.entries()) {
        assert.equal(Number(match[1]), index + 1, summary);
        if (match[2] !== 'undefined') assert.ok(Number.isFinite(Number(match[2])), summary);
      }
      assert.match(summary, /invalid\/blocked classifications 0\./);
      return {summary, finiteValues: probeValues.filter(match => match[2] !== 'undefined').length};
    });
  }

  await runCase('visibility-edge-budget-preserves-dense-metrics', async () => {
    await navigateToStep('synthetic-sightlines');
    await selectOption('view', 'visibility');
    await selectOption('visibilityDistance', '55');
    await selectOption('visibilityMetric', 'degree');
    await selectOption('visibilityEdgeCapacity', '8');
    const smallSummary = await waitForReadout(
      {view: 'visibility', visibilityDistance: '55', visibilityEdgeCapacity: '8', visibilityMetric: 'degree'},
      'blockVisibility',
      value => value.startsWith('Synthetic observers: 16; 8 / 52 edges in bounded drawn list (overflow 1);')
    );
    const smallValues = assertVisibilitySummary(smallSummary, 55, 'degree', 8);
    await selectOption('visibilityEdgeCapacity', '120');
    const fullSummary = await waitForReadout(
      {view: 'visibility', visibilityDistance: '55', visibilityEdgeCapacity: '120', visibilityMetric: 'degree'},
      'blockVisibility',
      value => value.startsWith('Synthetic observers: 16; 52 / 52 edges in bounded drawn list (overflow 0);')
    );
    const fullValues = assertVisibilitySummary(fullSummary, 55, 'degree', 120);
    assert.deepEqual(smallValues, fullValues, 'Edge-list capacity cannot change full-graph degree metrics');
    return {smallSummary, fullSummary, metricsUnchanged: true};
  });

  await navigateToStep('sampled-support');
  await selectOption('view', 'tessellation');
  await selectOption('inset', '0.4');
  await selectOption('sampleCapacity', '128');
  for (const spacing of ['4', '8', '16']) {
    await runCase(`tessellation-inset-0.4-spacing-${spacing}-capacity-128`, async () => {
      await selectOption('sampleSpacing', spacing);
      const expected = expectedSamples.get(`inset-0.4m-${spacing}-128`);
      const summary = await waitForReadout(
        {view: 'tessellation', inset: '0.4', sampleSpacing: spacing, sampleCapacity: '128'},
        'blockTessellation',
        value => value.startsWith(`Boundary-sampled Voronoi partition from GPUBufferSurface synthetic −0.4 m boundaries at ${spacing} m spacing with 128 site slots:`)
      );
      assert.ok(expected, `Independent sampling counts for ${spacing} m`);
      const counts = summary.match(/: (\d+) \/ (\d+) sites; sample overflow (\d+), invalid source segments (\d+); per-building completion flags ([\d, ]+); (.*)\. Areas are/);
      assert.ok(counts, summary);
      assert.equal(Number(counts[1]), expected.writtenSampleCount, summary);
      assert.equal(Number(counts[2]), expected.requiredSampleCount, summary);
      assert.equal(Number(counts[3]), 0, summary);
      assert.equal(Number(counts[4]), 0, summary);
      assert.ok(counts[5].split(',').every(flag => Number(flag.trim()) === 1), summary);
      assert.match(counts[6], /^complete$/);

      const titles = lastReadouts.chartTitles.blockTessellationChart ?? [];
      const areas = parseChartValues(titles);
      assert.equal(areas.length, 3, 'Completed tessellation must expose one area for each source building');
      areaReadoutAvailable = true;
      const displayedTotalArea = areas.reduce((sum, row) => sum + row.value, 0);
      assert.ok(Math.abs(displayedTotalArea - reference.enclosure.area) <= 3.5,
        `Visible cell-area total ${displayedTotalArea} should cover CPU enclosure area ${reference.enclosure.area}; ${JSON.stringify(titles)}`);
      return {summary, areaReadoutAvailable: areas.length === 3, areas};
    });
  }

  await runCase('tessellation-small-site-budget-withholds-incomplete-result', async () => {
    await selectOption('sampleSpacing', '8');
    await selectOption('sampleCapacity', '16');
    const expected = expectedSamples.get('inset-0.4m-8-16');
    const summary = await waitForReadout(
      {view: 'tessellation', inset: '0.4', sampleSpacing: '8', sampleCapacity: '16'},
      'blockTessellation',
      value => value.startsWith('Boundary-sampled Voronoi partition from GPUBufferSurface synthetic −0.4 m boundaries at 8 m spacing with 16 site slots:')
    );
    const counts = summary.match(/: (\d+) \/ (\d+) sites; sample overflow (\d+), invalid source segments (\d+); per-building completion flags ([\d, ]+); (.*)\. Areas are/);
    assert.ok(counts, summary);
    assert.ok(expected, 'Independent sampling counts for the 16-site budget');
    assert.equal(Number(counts[1]), expected.writtenSampleCount, summary);
    assert.equal(Number(counts[2]), expected.requiredSampleCount, summary);
    assert.ok(Number(counts[3]) > 0, summary);
    assert.match(counts[6], /incomplete|withheld/i);
    return {summary};
  });

  await runCase('tessellation-zero-inset-duplicate-sites-fail-honestly', async () => {
    const sourceDuplicates = reference.duplicateSites.find(row => row.geometry === 'source' && row.spacing === 8);
    assert.ok(sourceDuplicates.crossOwnerDuplicateSitePairCount > 0, 'Independent source boundary samples contain duplicate sites');
    await selectOption('inset', '0');
    await selectOption('sampleSpacing', '8');
    await selectOption('sampleCapacity', '128');
    const summary = await waitForReadout(
      {view: 'tessellation', inset: '0', sampleSpacing: '8', sampleCapacity: '128'},
      'blockTessellation',
      value => value.startsWith('Boundary-sampled Voronoi partition from GPUBufferSurface zero-distance boundaries at 8 m spacing with 128 site slots:')
    );
    assert.match(summary, /incomplete|withheld|invalid source segments [1-9]|completion flags (?:0|[\d, ]*0)/i);
    assert.doesNotMatch(summary, /; complete\. Areas are/);
    return {summary, independentDuplicateSitePairs: sourceDuplicates.crossOwnerDuplicateSitePairCount};
  });

  await runCase('tessellation-recovers-at-inset-0.4-capacity-128', async () => {
    await selectOption('inset', '0.4');
    await selectOption('sampleSpacing', '8');
    await selectOption('sampleCapacity', '128');
    const summary = await waitForReadout(
      {view: 'tessellation', inset: '0.4', sampleSpacing: '8', sampleCapacity: '128'},
      'blockTessellation',
      value => value.startsWith('Boundary-sampled Voronoi partition from GPUBufferSurface synthetic −0.4 m boundaries at 8 m spacing with 128 site slots:') && value.includes('; complete. Areas are')
    );
    return {summary};
  });

  await runCase('view-cache-and-query-changes-preserve-readouts', async () => {
    await navigateToStep('mapped-adjacency');
    await selectOption('view', 'adjacency');
    const adjacencyBefore = await waitForReadout(
      {view: 'adjacency'},
      'blockAdjacency',
      value => value.includes('GPU adjacency recipe complete;')
    );
    await navigateToStep('bounds-candidates');
    await selectOption('view', 'bounds');
    await selectOption('boundsRelation', 'intersects');
    await setRangeOption('queryHalfExtent', 48);
    const boundsAfterQuery = await waitForReadout(
      {view: 'bounds', boundsRelation: 'intersects', queryHalfExtent: '48'},
      'blockBounds',
      value => value.startsWith('GPUBoundsFilter intersects: 3 / 3 footprints in query;')
    );
    await navigateToStep('mapped-adjacency');
    await selectOption('view', 'adjacency');
    const adjacencyAfter = await waitForReadout(
      {view: 'adjacency'},
      'blockAdjacency',
      value => value.includes('GPU adjacency recipe complete;')
    );
    assert.equal(adjacencyAfter, adjacencyBefore, 'Changing query bounds and returning to cached adjacency preserves the summary');
    const adjacencyChart = parseChartValues(lastReadouts.chartTitles.blockAdjacencyChart ?? []);
    assert.equal(adjacencyChart.length, 3, 'Cached adjacency chart remains available');
    return {boundsAfterQuery, adjacencyPreserved: true, adjacencyChart};
  });
  }
} catch (error) {
  errors.push(`harness: ${String(error)}`);
} finally {
  lastBody = page ? await page.locator('body').innerText().catch(() => '') : '';
  if (browser) await browser.close().catch(error => errors.push(`browser close: ${String(error)}`));
  if (!areaReadoutAvailable && !smoke) {
    capabilities.unavailableDiagnostics.push('No tessellation area chart values were exposed in visible readouts; only sample counts and completeness are asserted.');
  }
  await writeFile(outputFile, JSON.stringify({
    baseUrl,
    reference,
    expectedCaseCount,
    completedCaseCount: reports.length,
    capabilities,
    areaReadoutAvailable,
    reports,
    errors,
    lastBody,
    lastReadouts
  }, null, 2) + '\n');
}

assert.deepEqual(errors, [], `Block relationships acceptance failed; report written to ${outputFile}`);
assert.equal(reports.length, expectedCaseCount, `Expected ${expectedCaseCount} acceptance cases; report written to ${outputFile}`);
assert.ok(reports.every(item => item.passed), `Block relationships cases failed; report written to ${outputFile}`);
console.log(`Accepted ${reports.length} block relationships cases; report: ${outputFile}`);
