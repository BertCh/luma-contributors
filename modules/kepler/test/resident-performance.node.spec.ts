// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  counterDelta,
  createInteractionRanges,
  createPerformanceColumns,
  expectedRangeCount,
  parsePerformanceOptions,
  summarizeDurations,
  summarizeCpuProfile,
  summarizeEncodeTelemetry
} from '../../../examples/deck/kepler-compute/scripts/resident-performance-utils.mjs';

test('performance configuration bounds workload and preserves a small independent built-in fixture', () => {
  const options = parsePerformanceOptions([
    '--rows',
    '1000000',
    '--iterations',
    '40',
    '--url',
    'http://localhost:5318/?rows=99&map=x'
  ]);
  expect(options.rows).toBe(1000000);
  expect(options.diagnostics).toBe(false);
  expect(options.interaction).toBe('create-or-update');
  expect(parsePerformanceOptions(['--interaction', 'value-only']).interaction).toBe('value-only');
  expect(parsePerformanceOptions(['--diagnostics']).diagnostics).toBe(true);
  expect(options.iterations).toBe(40);
  expect(new URL(options.url).searchParams.get('rows')).toBe('256');
  expect(new URL(options.url).searchParams.get('map')).toBe('x');
  for (const argumentsList of [
    ['--rows', '1000001'],
    ['--iterations', '0'],
    ['--iterations', '101'],
    ['--warmup', '0'],
    ['--batch-size', '1'],
    ['--timeout-ms', 'Infinity'],
    ['--url', 'file:///tmp/test'],
    ['--unknown'],
    ['--interaction', 'slider-ish'],
    ['--output', '--headed']
  ])
    expect(() => parsePerformanceOptions(argumentsList)).toThrow();
});

test('deterministic source columns and inclusive count oracle survive arbitrary batch seams', () => {
  const complete = createPerformanceColumns(0, 2327);
  const first = createPerformanceColumns(0, 1111);
  const second = createPerformanceColumns(1111, 1216);
  for (const column of ['longitude', 'latitude', 'value'] as const)
    expect([...first[column], ...second[column]]).toEqual([...complete[column]]);
  for (const [minimum, maximum] of [
    ...createInteractionRanges(100),
    [-10, 2000],
    [5.1, 5.9],
    [1001, 2000]
  ])
    expect(expectedRangeCount(complete.value.length, minimum, maximum)).toBe(
      complete.value.filter(value => value >= minimum && value <= maximum).length
    );
  const ranges = createInteractionRanges(100);
  for (let index = 1; index < ranges.length; index++)
    expect(expectedRangeCount(100000, ...ranges[index])).not.toBe(
      expectedRangeCount(100000, ...ranges[index - 1])
    );
});

test('latency summaries use actual nearest-rank observations without mutating samples', () => {
  const samples = [8, 1, 4, 2, 3];
  expect(summarizeDurations(samples)).toMatchObject({
    minimum: 1,
    p50: 3,
    p95: 8,
    maximum: 8,
    count: 5
  });
  expect(samples).toEqual([8, 1, 4, 2, 3]);
  expect(summarizeDurations([])).toMatchObject({available: false});
  expect(() => summarizeDurations([NaN])).toThrow();
  expect(() => summarizeDurations([-1])).toThrow();
});

test('transfer reports distinguish unavailable and reset counters from genuine zero transfers', () => {
  expect(counterDelta(undefined, undefined)).toMatchObject({available: false});
  expect(counterDelta(8, 8)).toEqual({available: true, before: 8, after: 8, delta: 0});
  expect(counterDelta(8, 4)).toMatchObject({
    available: false,
    reason: 'Counter reset during measurement'
  });
  expect(counterDelta({summary: 8}, {summary: 12, table: 4})).toMatchObject({
    summary: {available: true, delta: 4},
    table: {available: false}
  });
});

test('CPU diagnostics aggregate leaf identities without fabricating execution durations', () => {
  const profile = {
    startTime: 1000,
    endTime: 6000,
    nodes: [
      {
        id: 1,
        callFrame: {functionName: 'encode', url: '/runtime.ts', lineNumber: 4, columnNumber: 1}
      },
      {
        id: 2,
        callFrame: {functionName: 'encode', url: '/other.ts', lineNumber: 4, columnNumber: 1}
      },
      {id: 3, callFrame: {functionName: '', url: '', lineNumber: 0, columnNumber: 0}}
    ],
    samples: [1, 2, 1, 3]
  };
  const summary = summarizeCpuProfile(profile);
  expect(summary.sampleCount).toBe(4);
  expect(summary.observedProfileMilliseconds).toBe(5);
  expect(summary.topLeafFrames.map(frame => [frame.functionName, frame.samples])).toEqual([
    ['encode', 2],
    ['encode', 1],
    ['(anonymous)', 1]
  ]);
  expect(summary.topLeafFrames[0]).not.toHaveProperty('milliseconds');
  expect(summarizeCpuProfile({...profile, samples: []}).sampleCount).toBe(0);
  expect(() => summarizeCpuProfile({...profile, samples: [999]})).toThrow('unknown node');
});

test('cumulative encode diagnostics retain intermediate revisions and expose event truncation', () => {
  const before = {generation: 1, count: 5, totalMilliseconds: 20, droppedEvents: 0, events: []};
  const events = [
    {sequence: 6, sourceRevision: 1, explorerRevision: 11, startedAt: 100, finishedAt: 104},
    {sequence: 7, sourceRevision: 1, explorerRevision: 13, startedAt: 110, finishedAt: 119}
  ];
  const after = {...before, count: 7, totalMilliseconds: 33, events};
  expect(summarizeEncodeTelemetry(before, after, 1)).toMatchObject({
    available: true,
    count: 2,
    javascriptMilliseconds: 13,
    eventsComplete: true,
    events
  });
  expect(
    summarizeEncodeTelemetry(before, {...after, events: [events[1]], droppedEvents: 1}, 1)
  ).toMatchObject({
    count: 2,
    javascriptMilliseconds: 13,
    eventsComplete: false,
    droppedEvents: 1
  });
  expect(summarizeEncodeTelemetry(before, {...after, generation: 2}, 1)).toMatchObject({
    available: false
  });
  expect(summarizeEncodeTelemetry(before, {...after, count: 4}, 1)).toMatchObject({
    available: false
  });
  expect(summarizeEncodeTelemetry(before, after, 2)).toMatchObject({available: false});
  expect(summarizeEncodeTelemetry(null, after, 1)).toMatchObject({available: false});
});
