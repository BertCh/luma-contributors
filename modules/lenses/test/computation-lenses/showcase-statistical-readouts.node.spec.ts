// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, it} from 'vitest';
import {GPU_SPACE_TIME_SUMMARY} from '../../../analysis/src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {getLocalPooling} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/statistics/rate-smoothing.stats';
import {getKernelValue} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/weights/spatial-weights.charts';
import {resolveProfileStandardization} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/statistics/similar-places-standardization';
import {
  getReferenceProfile,
  getContributions,
  standardizeColumns
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/statistics/similar-places.stats';

const source = readFileSync(
  'examples/deck/spatial-analysis-showcase/src/scenes/time/space-time-clusters.compute.ts',
  'utf8'
);
const compile = (text: string) =>
  transpileModule(text, {compilerOptions: {target: ScriptTarget.ES2022}}).outputText;

it('Knox histogram uses the submitted count when a later slider value arrives first', () => {
  const marker = source.indexOf('      `${key}-knox`,');
  const start = source.indexOf('      bytes => {', marker);
  const end = source.indexOf('\n    );', start);
  const formatStart = source.indexOf('function formatDistribution(');
  const formatEnd = source.indexOf('\ntype SourceTable', formatStart);
  const readouts = new Map<string, unknown>();
  const callback = new Function(
    'ctx',
    'SUMMARY',
    `
    const destroyed = false, current = {key: 'fixture'}, key = 'fixture';
    const GPU_SPACE_TIME_SUMMARY_LENGTH = 12, GPU_SPACE_TIME_PARAMETER_LENGTH = 4, MAXIMUM_PERMUTATIONS = 999;
    const formatNumber = String, formatProbability = String, getKnoxPoissonPValue = () => NaN;
    ${compile(source.slice(formatStart, formatEnd))}
    ${compile(`const callback = ${source.slice(start, end)};`)} return callback;
  `
  )(
    {
      options: {permutations: 4},
      setReadout: (key: string, value: unknown) => readouts.set(key, value)
    },
    GPU_SPACE_TIME_SUMMARY
  );
  const words = new Uint32Array(24 + 1 + 2000 + 1 + 4);
  new Float32Array(words.buffer)[0] = 1;
  words.set([1, 0, 2, 0, 0], 25); // observation then two attempted draws, followed by tail slots
  words[words.length - 2] = 2; // copied parameter permutationCount, not live slider=4
  callback(words.buffer);
  expect(readouts.get('knoxDistribution')).toContain('50.0%');
});

it('canonical and saved z-score spellings publish the same hand-worked profile and units', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/statistics/similar-places.compute.ts',
    'utf8'
  );
  const start = source.indexOf('  const publishProfile =');
  const end = source.indexOf('  const publishAnnotations =', start);
  // [0,2,4] has mean 2, population deviation sqrt(8/3); endpoint z-scores are ±sqrt(3/2).
  const standardized = {
    zscore: standardizeColumns([[0, 2, 4]], 'zscore', 3),
    rank: standardizeColumns([[0, 2, 4]], 'rank', 3)
  };
  const charts = new Map<
    string,
    {rows: {a: number; b: number}[]; xDomain: number[]; xLabel: string; values: number[]}
  >();
  const options = {standardization: 'z-score'};
  const publish = new Function(
    'ctx',
    'standardized',
    'getReferenceProfile',
    'resolveProfileStandardization',
    `
    const references = [0], nameOf = String, SIMILARITY_ATTRIBUTES = [{label: 'Fixture'}];
    const formatCount = String, getRankOf = () => 1;
    ${compile(source.slice(start, end))}; return publishProfile;
  `
  )(
    {options, setChart: (key: string, value: never) => charts.set(key, value), setReadout() {}},
    standardized,
    getReferenceProfile,
    resolveProfileStandardization
  );
  const profile = getReferenceProfile(standardized.zscore, [0]);
  const contributions = getContributions(standardized.zscore, profile, [1], 2);
  expect(contributions[0].contribution).toBeCloseTo(6, 12);
  publish(contributions, 2);
  const canonical = charts.get('profile')!;
  expect(canonical.rows[0].a).toBeCloseTo(-1.224744871391589, 12);
  expect(canonical.rows[0].b).toBeCloseTo(1.224744871391589, 12);
  expect(canonical.xDomain).toEqual([-4, 4]);
  expect(canonical.xLabel).toContain('Standard deviations');
  expect(charts.get('contribution')!.values).toEqual([100]);
  options.standardization = 'rank';
  publish(
    getContributions(standardized.rank, getReferenceProfile(standardized.rank, [0]), [1], 2),
    2
  );
  expect(charts.get('profile')!.rows[0]).toMatchObject({a: 0, b: 1});
  expect(charts.get('profile')!.xDomain).toEqual([0, 1]);
  expect(charts.get('profile')!.xLabel).toContain('Percentile');
  const scene = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/statistics/similar-places.scene.ts',
    'utf8'
  );
  const note = scene
    .slice(scene.indexOf('    note: `Weighted distance on ') + '    note: '.length)
    .split('\n')[0];
  const getNote = new Function('state', 'resolveProfileStandardization', `return ${note};`);
  expect(getNote({standardization: 'z-score'}, resolveProfileStandardization)).toContain(
    'on z-scores'
  );
  expect(getNote({standardization: 'rank'}, resolveProfileStandardization)).toContain(
    'on percentile ranks'
  );
  expect(() => resolveProfileStandardization('zscore' as never)).toThrow(/z-score.*rank/);
  expect(resolveProfileStandardization('rank')).toBe('rank');
  expect(() => resolveProfileStandardization('typo' as never)).toThrow(/z-score.*rank/);
});

it('Mantel histogram uses finite null draws only inside the attempted prefix', () => {
  const start = source.indexOf('function formatDistribution(');
  const end = source.indexOf('\ntype SourceTable', start);
  const format = new Function(`${compile(source.slice(start, end))}; return formatDistribution;`)();
  // Two usable draws, one below r=.5: percentile is 1/2, not 1/3. Tail 99 is unattempted.
  const text = format(Float32Array.of(0, NaN, 1, 99), 3, 0.5);
  expect(text).toContain('50.0%');
  expect(text.split('  ')[0]).toContain('@');
  expect(format(Float32Array.of(NaN, NaN, 1), 2, 0.5)).toBe('unavailable');
});

it('scan scene suppresses invalid inference, recovers, and shares the float32 map cutoff', () => {
  const start = source.indexOf('  function showScan(');
  const end = source.indexOf('  function adoptVariant(', start);
  const readouts = new Map<string, unknown>();
  const legends: unknown[] = [];
  const options = {significance: 0.05};
  const reader = new Function(
    'ctx',
    `
    const GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH = 4, MAXIMUM_CLUSTERS = 7, zoneCount = 2, BUCKET_DAYS = 14;
    let cpuMembership, _cpuStatistics, llrMaximum;
    const formatCount = String, formatDay = String, formatProbability = String;
    ${compile(source.slice(start, end))}; return showScan;
  `
  )({
    options,
    setReadout: (key: string, value: unknown) => readouts.set(key, value),
    setLegendExtent: (...args: unknown[]) => legends.push(args),
    requestLayers() {}
  });
  const words = new Uint32Array(4 + 7 * 12 + 3 + 2);
  new Float32Array(words.buffer)[words.length - 1] = Math.fround(0.05);
  const statistics = new Float32Array(words.buffer, 16 + 7 * 16, 7 * 8);
  statistics.fill(NaN);
  words[words.length - 2] = 1;
  reader(words.buffer);
  expect(readouts.get('scanClusters')).toMatch(/unavailable/i);
  expect(legends).toHaveLength(0);
  words[words.length - 2] = 0;
  reader(words.buffer);
  expect(readouts.get('scanClusters')).toBe('0 found, 0 significant');
  expect(legends).toHaveLength(0);
  words[0] = 1; // Numerical loss with no diagnostic word still cannot publish NaN as LLR.
  reader(words.buffer);
  expect(readouts.get('scanClusters')).toMatch(/unavailable/i);
  expect(legends).toHaveLength(0);
  words[0] = 2;
  statistics.fill(0);
  statistics[0] = 2;
  statistics[3] = Math.fround(1 / 20); // 19 draws: attainable floor agrees with shader alpha=.05f.
  statistics[8] = 1;
  statistics[11] = Math.fround(0.1); // Genuinely nonsignificant control.
  reader(words.buffer);
  expect(readouts.get('scanClusters')).toBe('2 found, 1 significant');
  expect(legends.at(-1)).toEqual(['llr', [0, 2]]);
  options.significance = 0.01; // Slider moved after submission; these bytes still belong to alpha=.05.
  reader(words.buffer);
  expect(readouts.get('scanClusters')).toBe('2 found, 1 significant');
});

it('tooltip pooling excludes negative focal and donor events in both moment passes', () => {
  // Row 1 is ineligible. Row 0 pools only its 6 events / 30 woman-years = .2.
  const result = getLocalPooling([6, -2], [30, 10], [0, 1, 2], [1, 0]);
  expect(Number.isFinite(result.pooledRates[0])).toBe(true);
  expect(result.pooledRates[0]).toBeCloseTo(0.2, 7);
  expect(result.pooledRates[1]).toBeNaN();
  expect(result.weights[0]).toBe(0);
  expect(result.weights[1]).toBeNaN();
  expect(Array.from(result.eligibleNeighborCounts)).toEqual([0, 0]);
  for (const invalid of [NaN, Infinity, -1]) {
    expect(getLocalPooling([6, invalid], [30, 10], [0, 1, 2], [1, 0]).pooledRates[0]).toBeCloseTo(
      0.2,
      7
    );
  }
  // Eligible exposure 2+2=4, pooled events 8 => m=2. Weighted variance=4,
  // noise correction m/(B/n)=1, prior variance=3, own noise=1 => weight=3/4.
  const counted = getLocalPooling([8, 0, -1], [2, 2, 8], [0, 2, 2, 2], [1, 2]);
  expect(counted.pooledRates[0]).toBe(2);
  expect(counted.weights[0]).toBe(0.75);
  expect(counted.eligibleNeighborCounts[0]).toBe(1);
});

it('quartic chart uses the bisquare profile rather than the uniform profile', () => {
  // (15/16)*(1 - (1/4)^2)^2 = 3375/4096.
  expect(getKernelValue('quartic', 0.25)).toBe(0.823974609375);
  expect(getKernelValue('uniform', 0.25)).toBe(0.5);
  expect(() => getKernelValue('typo' as never, 0.25)).toThrow(
    /gaussian.*triangular.*epanechnikov.*bisquare.*quartic.*uniform/
  );
});
