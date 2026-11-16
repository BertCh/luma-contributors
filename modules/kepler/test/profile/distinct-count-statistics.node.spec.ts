// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  KEPLER_PROFILE_MAXIMUM_REFINED_DISTINCT_COUNT,
  KeplerDistinctCounter,
  getKeplerDistinctCountExactLimit
} from '../../src/profile/distinct-count-statistics';
import {
  profileKeplerColumnTable,
  profileKeplerFeatureCollection
} from '../../src/profile/dataset-profile';
import {evaluateGPUAnalysisRequirement} from '@luma-contributors/analysis/gpu-spatial-analysis';

const PROFILE_OPTIONS = {datasetId: 'bounded-distinct', label: 'Bounded distinct'};

function evaluateDistinctCapacity(values: readonly unknown[], limit: number, refine = false) {
  const profile = profileKeplerColumnTable(
    {rowCount: values.length, columns: [{name: 'category', values}]},
    {...PROFILE_OPTIONS, ...(refine ? {distinctCountThresholds: [limit]} : {})}
  );
  const result = evaluateGPUAnalysisRequirement(
    {
      id: 'maximum-distinct-values',
      role: 'category',
      severity: 'block',
      limit,
      remedy: 'Aggregate categories'
    },
    profile,
    {category: 'category'}
  );
  return {column: profile.columns[0], result};
}

describe('deterministic distinct-count evidence', () => {
  it('is exact through the set limit and spills on the next distinct value', () => {
    const counter = new KeplerDistinctCounter(3, 32);
    for (const value of [0, -0, 1, 1, 2]) counter.add(value);
    expect(counter.getStatistics()).toEqual({count: 3, exact: true, lowerBound: 3, upperBound: 3});
    counter.add(3);
    expect(counter.isExact()).toBe(false);
    expect(counter.getStatistics()).toMatchObject({count: 4, lowerBound: 4, upperBound: 4});
    counter.add(3);
    // After spilling, even equal hashes cannot prove equality of the underlying values.
    expect(counter.getStatistics()).toMatchObject({lowerBound: 4, upperBound: 5});
  });

  it('keeps conservative intervals under collisions, saturation and repeated values', () => {
    const counter = new KeplerDistinctCounter(2, 32);
    const oracle = new Set<number | string>();
    // More distinct keys than bits force collisions; sequential mixed values saturate all 32 bits.
    for (let index = 0; index < 2000; index++) {
      const value = index % 3 === 0 ? `category-${index % 401}` : index % 307;
      oracle.add(value);
      counter.add(value);
      const statistics = counter.getStatistics();
      expect(statistics.lowerBound).toBeLessThanOrEqual(oracle.size);
      expect(statistics.upperBound).toBeGreaterThanOrEqual(oracle.size);
      expect(statistics.count).toBeGreaterThanOrEqual(statistics.lowerBound);
      expect(statistics.count).toBeLessThanOrEqual(statistics.upperBound);
    }
    expect(counter.getStatistics().lowerBound).toBe(32);
    expect(counter.getCount()).toBe(counter.getStatistics().upperBound);
  });

  it('refines near a requested threshold and refuses to promise unbounded exact storage', () => {
    expect(getKeplerDistinctCountExactLimit([10000, 15000])).toBe(15001);
    expect(getKeplerDistinctCountExactLimit([Number.MAX_SAFE_INTEGER])).toBe(
      KEPLER_PROFILE_MAXIMUM_REFINED_DISTINCT_COUNT
    );
    for (const threshold of [-1, NaN, 1.5]) {
      expect(() => getKeplerDistinctCountExactLimit([threshold])).toThrow(
        /KeplerDistinctCounter refinementThreshold/
      );
    }
    expect(() => new KeplerDistinctCounter(0)).toThrow(/exactCountLimit/);
    expect(() => new KeplerDistinctCounter(10, 33)).toThrow(/bitmapBitCount/);
  });

  it('settles hard category capacity only from guaranteed evidence, including exact refinement', () => {
    // Every key occurs twice after the spill: the bitmap alone cannot prove those repeats.
    const values = Array.from({length: 30000}, (_, index) => `category-${index % 15000}`);
    const ordinary = evaluateDistinctCapacity(values, 15000);
    expect(ordinary.column.distinctCountIsEstimate).toBe(true);
    expect(ordinary.column.distinctCountLowerBound).toBeLessThanOrEqual(15000);
    expect(ordinary.column.distinctCountUpperBound).toBeGreaterThan(15000);
    expect(ordinary.result.state).toBe('unknown');
    expect(ordinary.result.message).toContain('refine with an exact count');
    const refined = evaluateDistinctCapacity(values, 15000, true);
    expect(refined.column).toMatchObject({
      distinctCount: 15000,
      distinctCountLowerBound: 15000,
      distinctCountUpperBound: 15000
    });
    expect(refined.column.distinctCountIsEstimate).toBeUndefined();
    expect(refined.result.state).toBe('passed');
    expect(evaluateDistinctCapacity(values, 14999, true).result.state).toBe('failed');
    expect(evaluateDistinctCapacity(values, 100).result).toMatchObject({
      state: 'failed',
      remedy: 'Aggregate categories'
    });
    expect(evaluateDistinctCapacity(values, values.length).result.state).toBe('passed');
  });

  it('retains exact accepted-value semantics during feature-profile refinement', () => {
    const profile = profileKeplerFeatureCollection(
      {
        type: 'FeatureCollection',
        features: Array.from({length: 24000}, (_, index) => ({
          type: 'Feature',
          geometry: null,
          properties: {category: index % 12000}
        }))
      },
      {...PROFILE_OPTIONS, distinctCountThresholds: [12000]}
    );
    expect(profile.columns[0]).toMatchObject({
      distinctCount: 12000,
      distinctCountLowerBound: 12000,
      distinctCountUpperBound: 12000
    });
  });

  it('does not authorize whole-population capacity from a sample with few categories', () => {
    const values = Array.from({length: 100}, (_, index) =>
      index % 2 ? `unique-${index}` : 'common'
    );
    const profile = profileKeplerColumnTable(
      {rowCount: values.length, columns: [{name: 'category', values}]},
      {...PROFILE_OPTIONS, maximumSampleSize: 10, distinctCountThresholds: [2]}
    );
    const column = profile.columns[0];
    const actualCount = new Set(values).size;
    expect(column.distinctCountLowerBound).toBeLessThanOrEqual(actualCount);
    expect((column.distinctCountUpperBound as number) + 90).toBeGreaterThanOrEqual(actualCount);
    const evaluate = (limit: number) =>
      evaluateGPUAnalysisRequirement(
        {id: 'maximum-distinct-values', role: 'category', severity: 'block', limit},
        profile,
        {category: 'category'}
      );
    expect(evaluate(2)).toMatchObject({state: 'unknown', evidence: 'sampled-profile'});
    expect(evaluate(100)).toMatchObject({state: 'passed', evidence: 'sampled-profile'});
  });
});
