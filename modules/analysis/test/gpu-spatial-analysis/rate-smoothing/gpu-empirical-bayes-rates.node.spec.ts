// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/index';
import {computeEmpiricalBayesOracle} from './empirical-bayes-oracle';
import {EB_LOW_DISPERSION} from './eb-low-dispersion-fixture';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createRateAuditInput,
  RATE_AUDIT_EXPECTED
} from './rate-smoothing-mathematical-audit-fixture';
import {computeSpatialEmpiricalBayesOracle} from './spatial-empirical-bayes-oracle';

it('R8 default clamping and inactive zero-weight slots agree with the contributors', () => {
  expect(computeEmpiricalBayesOracle({events: [0], populations: [10]}).smoothed).toEqual([0]);
  const input = {
    events: [0, 20],
    populations: [10, 10],
    offsets: [0, 1, 2],
    neighbors: [1, 0],
    weights: [0, 0]
  };
  const spatial = computeSpatialEmpiricalBayesOracle(input);
  expect(spatial.spatialRate).toEqual([0, 2]);
  expect(spatial.smoothed).toEqual([0, 2]);
  expect(spatial.priorMean).toEqual([0, 2]);
  expect(spatial.priorVariance).toEqual([0, 0]);
});

it('GPUEmpiricalBayesRates validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'eb-validation'});
  const events = createTransientView(graph, 'events', 'float32', 10);
  const populations = createTransientView(graph, 'populations', 'float32', 10);
  expect(() => new GPUEmpiricalBayesRates({events, populations, output: {}})).toThrow(/no output/);
  expect(
    () =>
      new GPUEmpiricalBayesRates({
        events,
        populations: createTransientView(graph, 'short', 'float32', 9),
        output: {rawRates: createTransientView(graph, 'raw', 'float32', 10)}
      })
  ).toThrow(/populations length/);
  const nodes = new GPUEmpiricalBayesRates({
    events,
    populations,
    output: {standardizedRates: createTransientView(graph, 'z', 'float32', 10)}
  }).getCommandNodes(graph);
  // Five rate stages plus the exact uint32 count reduction.
  expect(nodes.length).toBe(7);
});

it('the empirical-Bayes oracle reproduces esda unclamped and the numpy Marshall clamp', () => {
  const input = {events: EB_LOW_DISPERSION.events, populations: EB_LOW_DISPERSION.populations};
  const esda = computeEmpiricalBayesOracle({...input, clampPriorVariance: false});
  const clamped = computeEmpiricalBayesOracle({...input, clampPriorVariance: true});
  expect(esda.priorVariance).toBeCloseTo(EB_LOW_DISPERSION.priorVariance, 12);
  for (const [row, expected] of EB_LOW_DISPERSION.unclamped.entries()) {
    expect(esda.smoothed[row]).toBeCloseTo(expected, 12);
  }
  for (const [row, expected] of EB_LOW_DISPERSION.clamped.entries()) {
    expect(clamped.smoothed[row]).toBeCloseTo(expected, 12);
  }
  expect(esda.smoothed[0]).toBeLessThan(0);
});

it('GPUEmpiricalBayesRates builds the same nodes with and without the clamp', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'eb-clamp'});
  const events = createTransientView(graph, 'events', 'float32', 10);
  const populations = createTransientView(graph, 'populations', 'float32', 10);
  for (const clampPriorVariance of [true, false, undefined]) {
    const nodes = new GPUEmpiricalBayesRates({
      id: `eb-${clampPriorVariance}`,
      events,
      populations,
      clampPriorVariance,
      output: {smoothedRates: createTransientView(graph, `s-${clampPriorVariance}`, 'float32', 10)}
    }).getCommandNodes(graph);
    expect(nodes.length).toBe(7);
  }
});

it('GPUEmpiricalBayesRates accepts uint32 counts with one conversion node per input', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'eb-uint32'});
  const nodes = new GPUEmpiricalBayesRates({
    events: createTransientView(graph, 'events', 'uint32', 10),
    populations: createTransientView(graph, 'populations', 'uint32', 10),
    output: {smoothedRates: createTransientView(graph, 'smoothed', 'float32', 10)}
  }).getCommandNodes(graph);
  expect(nodes.length).toBe(9);
  expect(nodes.slice(0, 2).map(node => node.id)).toEqual([
    'empirical-bayes-rates-events-to-float32',
    'empirical-bayes-rates-populations-to-float32'
  ]);
  expect(
    () =>
      new GPUEmpiricalBayesRates({
        events: createTransientView(graph, 'bad-events', 'sint32', 10) as never,
        populations: createTransientView(graph, 'populations-2', 'float32', 10),
        output: {smoothedRates: createTransientView(graph, 'smoothed-2', 'float32', 10)}
      })
  ).toThrow(/events/);
});

it('the clamped empirical-Bayes CPU twin keeps all-zero events at zero', () => {
  // m=a=0: the documented weight is 0 and the smoothed rate is 0, not 0/0.
  const actual = computeEmpiricalBayesOracle({
    events: [0],
    populations: [10],
    clampPriorVariance: true
  });
  expect(actual.smoothed).toEqual([0]);
  expect(actual.raw).toEqual([0]);
  expect(actual.standardized[0]).toBeNaN();
});

it('both rate CPU twins match independent NumPy moments on every numerical output', () => {
  const input = createRateAuditInput();
  const global = computeEmpiricalBayesOracle({...input, clampPriorVariance: true});
  const spatial = computeSpatialEmpiricalBayesOracle(input);
  const differences: Record<string, number> = {};
  for (const [prefix, actual, expected] of [
    ['global', global, RATE_AUDIT_EXPECTED.global],
    ['spatial', spatial, RATE_AUDIT_EXPECTED.spatial]
  ] as const) {
    for (const name of Object.keys(expected)) {
      if (name === 'summary' || name === 'excess') continue;
      const actualValues = (actual as unknown as Record<string, number[]>)[name];
      const expectedValues = (expected as unknown as Record<string, readonly (number | null)[]>)[
        name
      ];
      let maximum = 0;
      for (const [row, value] of expectedValues.entries()) {
        if (value === null) expect(actualValues[row]).toBeNaN();
        else {
          expect(Number.isFinite(actualValues[row])).toBe(true);
          maximum = Math.max(maximum, Math.abs(actualValues[row] - value));
        }
      }
      expect(maximum).toBeLessThanOrEqual(2e-14);
      differences[`${prefix}.${name}`] = maximum;
    }
  }
  expect(global.count).toBe(6);
  expect(global.pooledRate).toBeCloseTo(RATE_AUDIT_EXPECTED.global.summary[3], 14);
  expect(global.weightedRateVariance).toBeCloseTo(RATE_AUDIT_EXPECTED.global.summary[4], 14);
  expect(global.priorVariance).toBeCloseTo(RATE_AUDIT_EXPECTED.global.summary[5], 14);
  console.info('Rate audit maximum absolute differences', differences);
});

function expectSmoothedRates(actual: number[], expected: number[]): void {
  expect(actual).toHaveLength(expected.length);
  for (const [row, value] of expected.entries()) {
    if (Number.isNaN(value)) expect(actual[row]).toBeNaN();
    else {
      expect(Number.isFinite(actual[row])).toBe(true);
      expect(actual[row]).toBeCloseTo(value, 12);
    }
  }
}

it('UST-03 excludes negative event counts from both priors and the focal output', () => {
  // Removing the negative sentinel leaves three identical rates: 2/10 = 0.2.
  const input = {events: [-10, 2, 2, 2], populations: [10, 10, 10, 10]};
  const global = computeEmpiricalBayesOracle(input);
  const spatial = computeSpatialEmpiricalBayesOracle({
    ...input,
    offsets: [0, 1, 3, 5, 6],
    neighbors: [1, 0, 2, 1, 3, 2]
  });
  expect(global.count).toBe(3);
  for (const smoothed of [global.smoothed, spatial.smoothed]) {
    expectSmoothedRates(smoothed, [NaN, 0.2, 0.2, 0.2]);
  }
});

it('DST-03 distinguishes the two prior variance conventions with hand-derived rates', () => {
  for (const clampPriorVariance of [true, false]) {
    const result = computeEmpiricalBayesOracle({
      events: [10, 11, 9],
      populations: [100, 100, 100],
      clampPriorVariance
    });
    // m=.1, a=-.0009333333; unclamped shrinkage weight is -14.
    const expected = clampPriorVariance ? [0.1, 0.1, 0.1] : [0.1, -0.04, 0.24];
    expectSmoothedRates(result.smoothed, expected);
  }
});

it('S4 rate expectations reject truncated and extra output arrays', () => {
  for (const invalid of [[NaN], [], [NaN, 0.2, 0.2, 0.2, 0.2]]) {
    expect(() => expectSmoothedRates(invalid, [NaN, 0.2, 0.2, 0.2])).toThrow();
  }
  for (const invalid of [[0.1], [], [0.1, -0.04, 0.24, 0.24]]) {
    expect(() => expectSmoothedRates(invalid, [0.1, -0.04, 0.24])).toThrow();
  }
});
