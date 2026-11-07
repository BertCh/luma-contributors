// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getBenchmarkDistribution, getBenchmarkSettings} from './benchmark-settings';

it('benchmark provenance is absent unless supplied and contradictory snapshots fail', () => {
  expect(getBenchmarkSettings().sourceSnapshot).toBeNull();
  const snapshot = {
    revision: 'a'.repeat(40),
    sha256: 'b'.repeat(64),
    capturedAt: '2026-10-24T00:00:00.000Z',
    fileCount: 10
  };
  expect(
    getBenchmarkSettings({VITE_BENCHMARK_SOURCE_SNAPSHOT: JSON.stringify(snapshot)}).sourceSnapshot
  ).toEqual(snapshot);
  expect(() =>
    getBenchmarkSettings({
      VITE_BENCHMARK_SOURCE_SNAPSHOT: JSON.stringify(snapshot),
      VITE_BENCHMARK_REVISION: 'c'.repeat(40)
    })
  ).toThrow(/Contradictory/);
  expect(() => getBenchmarkSettings({VITE_BENCHMARK_SOURCE_SNAPSHOT: '{}'})).toThrow();
});

it('benchmark sampling rejects empty measurements and reports even sample medians', () => {
  expect(() => getBenchmarkSettings({VITE_BENCHMARK_SAMPLES: '0'})).toThrow();
  expect(() => getBenchmarkSettings({VITE_BENCHMARK_WARMUPS: '-1'})).toThrow();
  expect(() => getBenchmarkSettings({VITE_BENCHMARK_SAMPLES: '1.5'})).toThrow();
  expect(getBenchmarkSettings({VITE_BENCHMARK_WARMUPS: '0'}).warmupCount).toBe(0);
  expect(getBenchmarkDistribution([5, 1, 2, 4])).toEqual({minimum: 1, median: 3, maximum: 5});
  expect(getBenchmarkDistribution([5, 1, 2])).toEqual({minimum: 1, median: 2, maximum: 5});
  expect(() => getBenchmarkDistribution([])).toThrow();
});
