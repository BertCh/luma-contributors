// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fileURLToPath} from 'node:url';
import {expect, test} from 'vitest';
import {
  classifyHardwareTier,
  evaluateAcceptance,
  loadAcceptanceTargets,
  parseAcceptanceOptions,
  renderAcceptanceMarkdown,
  summarizeSamples,
  summarizeVerdict
} from '../../../examples/deck/kepler-compute/scripts/acceptance-utils.mjs';

const targetsPath = fileURLToPath(
  new URL('../../../examples/deck/kepler-compute/scripts/acceptance-targets.json', import.meta.url)
);
const configuration = loadAcceptanceTargets(targetsPath);
const GIBIBYTE = 1073741824;
const bigLimits = {maxBufferSize: 4 * GIBIBYTE, maxStorageBufferBindingSize: 4 * GIBIBYTE};

function createReport(
  adapterInfo: Record<string, unknown>,
  limits = bigLimits,
  userAgent = 'Chrome'
) {
  return {
    adapterInfo,
    limits,
    features: [],
    userAgent,
    deviceMemoryGigabytes: 8,
    hardwareConcurrency: 8
  };
}

test('option parsing applies defaults and enforces bounds', () => {
  const options = parseAcceptanceOptions([]);
  expect(options.sizes).toEqual([1000000]);
  expect(options.iterations).toBe(20);
  expect(options.warmup).toBe(3);
  expect(options.frameSeconds).toBe(5);
  expect(options.timeoutMilliseconds).toBe(300000);
  expect(options.output.endsWith('acceptance-report.json')).toBe(true);
  expect(options.markdownOutput.endsWith('acceptance-report.md')).toBe(true);
  expect(options.targets.endsWith('acceptance-targets.json')).toBe(true);
  const custom = parseAcceptanceOptions([
    '--sizes',
    '1000000,5000000,10000000',
    '--warmup',
    '0',
    '--tier',
    'integrated',
    '--headed',
    '--label',
    'x'
  ]);
  expect(custom.sizes).toEqual([1000000, 5000000, 10000000]);
  expect(custom.warmup).toBe(0);
  expect(custom.tierOverride).toBe('integrated');
  expect(custom.headed).toBe(true);
  for (const argumentsList of [
    ['--sizes', '123'],
    ['--iterations', '0'],
    ['--iterations', '101'],
    ['--warmup', '21'],
    ['--frame-seconds', '61'],
    ['--timeout-ms', 'NaN'],
    ['--tier', 'nope'],
    ['--url', 'file:///x'],
    ['--bogus', '1'],
    ['--iterations']
  ]) {
    expect(() => parseAcceptanceOptions(argumentsList)).toThrow();
  }
});

test('hardware classification covers discrete, Apple, integrated, software and empty info', () => {
  const classify = (report: ReturnType<typeof createReport>) =>
    classifyHardwareTier(report, configuration.tiers);
  expect(
    classify(createReport({vendor: 'apple', architecture: 'metal-3', description: 'Apple M3 Pro'}))
      .tier
  ).toBe('reference-desktop');
  expect(
    classify(createReport({vendor: 'nvidia', architecture: 'ampere', description: 'RTX 3080'})).tier
  ).toBe('reference-desktop');
  const integrated = classify(
    createReport({vendor: 'intel', architecture: 'gen-12lp', description: 'Intel UHD Graphics'})
  );
  expect(integrated.tier).toBe('integrated');
  expect(integrated.reasons.length).toBeGreaterThan(0);
  expect(
    classify(
      createReport({vendor: 'google', architecture: 'swiftshader', description: 'SwiftShader'})
    ).tier
  ).toBe('low-power');
  expect(classify(createReport({vendor: 'nvidia'}, bigLimits, 'Android Mobile')).tier).toBe(
    'low-power'
  );
  expect(classify(createReport({}, {})).tier).toBe('unclassified');
  expect(classify(createReport({vendor: 'nvidia'}, {maxBufferSize: 1000})).tier).not.toBe(
    'reference-desktop'
  );
});

test('summarizeSamples uses nearest-rank percentiles', () => {
  expect(summarizeSamples([])).toBeNull();
  const stats = summarizeSamples(Array.from({length: 20}, (_, index) => 20 - index));
  expect(stats).toMatchObject({count: 20, p50: 10, p95: 19, min: 1, max: 20, mean: 10.5});
});

test('evaluation distinguishes pass, fail, unavailable, and blocked', () => {
  const stats = (p95: number) => ({count: 20, p50: 1, p95, min: 1, max: p95, mean: 2});
  const evaluate = (metrics: Record<string, unknown>) =>
    evaluateAcceptance({
      tier: 'reference-desktop',
      rows: 1000000,
      metrics,
      targets: configuration.targets
    });
  const passing = evaluate({filterLatency: stats(49.9)});
  const filter = (evaluation: ReturnType<typeof evaluate>) =>
    evaluation.results.find((result: {metric: string}) => result.metric === 'filterLatency');
  expect(filter(passing).status).toBe('pass');
  expect(filter(passing).origin).toBe('plan');
  expect(filter(evaluate({filterLatency: stats(50.1)})).status).toBe('fail');
  expect(evaluate({filterLatency: stats(50.1)}).verdict).toBe('fail');
  const unavailable = evaluate({filterLatency: null, warmUploadBytes: {bytes: 0}});
  expect(filter(unavailable).status).toBe('unavailable');
  expect(filter(unavailable).observed).toBeNull();
  expect(
    unavailable.results.find((result: {metric: string}) => result.metric === 'warmUploadBytes')
      .status
  ).toBe('pass');
  expect(unavailable.verdict).toBe('incomplete');
  const blocked = evaluate({blocked: {reason: 'admission failed'}});
  expect(blocked.results.every((result: {status: string}) => result.status === 'blocked')).toBe(
    true
  );
  expect(blocked.verdict).toBe('incomplete');
  const missing = evaluateAcceptance({
    tier: 'reference-desktop',
    rows: 42,
    metrics: {},
    targets: configuration.targets
  });
  expect(missing.results).toEqual([]);
  expect(missing.missingTargets.length).toBeGreaterThan(0);
  const full = evaluate({
    filterLatency: stats(10),
    brushLatency: stats(10),
    histogramLatency: stats(10),
    styleLatency: stats(10),
    linkedSummaryLatency: stats(10),
    frameTime: stats(10),
    uploadTime: {milliseconds: 100},
    memoryAccounted: {accountedBytes: 1000},
    warmUploadBytes: {bytes: 0},
    warmReadbackBytes: {bytes: 4096}
  });
  expect(full.verdict).toBe('pass');
  expect(summarizeVerdict([{evaluation: full}, {evaluation: blocked}])).toBe('incomplete');
  expect(summarizeVerdict([{evaluation: full}, {evaluation: passing}])).toBe('incomplete');
  expect(
    summarizeVerdict([{evaluation: full}, {evaluation: evaluate({filterLatency: stats(99)})}])
  ).toBe('fail');
});

test('targets file validates and only the documented targets are plan-origin', () => {
  expect(configuration.sizes).toEqual([1000000, 5000000, 10000000]);
  expect(configuration.tiers.at(-1).id).toBe('unclassified');
  const planEntries: string[] = [];
  for (const [tier, byRows] of Object.entries<any>(configuration.targets)) {
    for (const [rows, byMetric] of Object.entries<any>(byRows)) {
      for (const [metric, spec] of Object.entries<any>(byMetric)) {
        if (spec.origin === 'plan') planEntries.push(`${tier}/${rows}/${metric}`);
      }
    }
  }
  const expected = [
    'reference-desktop/1000000/filterLatency',
    'reference-desktop/1000000/linkedSummaryLatency',
    'reference-desktop/1000000/frameTime'
  ];
  for (const tier of configuration.tiers) {
    for (const rows of configuration.sizes) expected.push(`${tier.id}/${rows}/warmUploadBytes`);
  }
  expect(planEntries.sort()).toEqual(expected.sort());
  const reference = configuration.targets['reference-desktop']['1000000'];
  expect(reference.filterLatency).toMatchObject({statistic: 'p95', maxValue: 50});
  expect(reference.linkedSummaryLatency.maxValue).toBe(100);
  expect(reference.frameTime.maxValue).toBe(16.7);
  expect(reference.memoryAccounted.maxValue).toBe(536870912);
  expect(() => loadAcceptanceTargets(fileURLToPath(import.meta.url))).toThrow();
});

test('markdown reports verdict and disclaimers', () => {
  const evaluation = evaluateAcceptance({
    tier: 'reference-desktop',
    rows: 1000000,
    metrics: {blocked: {reason: 'admission failed'}},
    targets: configuration.targets
  });
  const markdown = renderAcceptanceMarkdown({
    schemaVersion: 1,
    label: 'test',
    generatedAt: '2026-10-30T00:00:00Z',
    environment: {
      browser: 'Chrome',
      deviceReport: createReport({vendor: 'apple'}),
      tier: 'reference-desktop',
      tierReasons: ['vendor matches apple'],
      gitRevision: 'abc'
    },
    runs: [{rows: 1000000, evaluation}]
  });
  expect(markdown).toContain('Overall verdict');
  expect(markdown).toContain('incomplete');
  expect(markdown).toContain('admission failed');
  expect(markdown).toContain('not GPU completion or physical presentation latency');
  expect(markdown).toContain('ratified by the project owner');
});
