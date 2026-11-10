// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUSpaceTimeHawkesParameterValues,
  GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH,
  GPUSpaceTimeHawkes,
  type GPUSpaceTimeHawkesProps
} from '../../../src/gpu-spatial-analysis/space-time-hawkes/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  fitHawkesReference,
  simulateHawkes,
  type HawkesEvent,
  type HawkesWindow
} from './stm08-hawkes-reference';

let viewCount = 0;

function createContributor(
  graph: GPUCommandGraph,
  rowCount: number,
  overrides: Partial<GPUSpaceTimeHawkesProps> = {},
  outputs: 'all' | 'minimal' = 'all'
) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `hawkes-node-view-${viewCount++}`, format, length);
  return new GPUSpaceTimeHawkes({
    id: overrides.id ?? `hawkes-${viewCount++}`,
    positions: createTransientView(graph, `hawkes-node-view-${viewCount++}`, 'float32x2', rowCount),
    times: view('float32', rowCount),
    parameters: view('float32', GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH),
    output:
      outputs === 'minimal'
        ? {estimates: view('float32', 4)}
        : {
            estimates: view('float32', 4),
            logLikelihood: view('float32', 1),
            backgroundProbability: view('float32', rowCount),
            status: {
              converged: view('uint32', 1),
              iterationLimitReached: view('uint32', 1),
              iterationCount: view('uint32', 1),
              candidateOverflow: view('uint32', 1),
              invalidCount: view('uint32', 1),
              approximation: view('uint32', 1)
            }
          },
    ...overrides
  });
}

it('GPUSpaceTimeHawkes estimates its command node count as an upper bound', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hawkes-nodes'});
  for (const [rowCount, maximumIterations, outputs] of [
    [100, 1, 'minimal'],
    [5000, 10, 'all'],
    [300000, 100, 'all'],
    [1000, 250, 'minimal']
  ] as const) {
    const contributor = createContributor(graph, rowCount, {maximumIterations}, outputs);
    const nodeCount = contributor.getCommandNodes(graph).length;
    expect(nodeCount).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    // The unrolled iterations are three nodes each; the rest is the fixed prologue and the sort.
    const withoutIterations = createContributor(graph, rowCount, {maximumIterations: 1}, outputs);
    expect(nodeCount - withoutIterations.getCommandNodes(graph).length).toBe(
      3 * (maximumIterations - 1)
    );
  }
});

it('GPUSpaceTimeHawkes rejects out-of-range capacities and shapes with named limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hawkes-limits'});
  expect(() => createContributor(graph, 100, {id: 'hawkes', maximumIterations: 0})).toThrow(
    'hawkes maximumIterations must be an integer in [1, 1000]'
  );
  expect(() => createContributor(graph, 100, {id: 'hawkes', maximumIterations: 1001})).toThrow(
    'hawkes maximumIterations must be an integer in [1, 1000]'
  );
  expect(() => createContributor(graph, 100, {id: 'hawkes', maximumParents: 0})).toThrow(
    'hawkes maximumParents must be an integer in [1, 4096]'
  );
  expect(() => createContributor(graph, 100, {id: 'hawkes', maximumParents: 4097})).toThrow(
    'hawkes maximumParents must be an integer in [1, 4096]'
  );
  expect(() => createContributor(graph, 1, {id: 'hawkes'})).toThrow(
    'hawkes needs at least two events'
  );
  expect(() =>
    createContributor(graph, 100, {
      id: 'hawkes',
      times: createTransientView(graph, 'hawkes-short-times', 'float32', 99)
    })
  ).toThrow('hawkes times length must equal positions length');
  expect(() =>
    createContributor(graph, 100, {
      id: 'hawkes',
      parameters: createTransientView(graph, 'hawkes-short-parameters', 'float32', 8)
    })
  ).toThrow('hawkes parameters must hold 9 float32 values');
  expect(() => createContributor(graph, 2 ** 24 + 1, {id: 'hawkes'})).toThrow(
    'hawkes positions length must be at most 16777216'
  );
});

it('getGPUSpaceTimeHawkesParameterValues packs the documented layout and rejects bad values', () => {
  const values = getGPUSpaceTimeHawkesParameterValues({
    windowArea: 4,
    windowDuration: 365,
    maximumLag: 30,
    maximumDistance: 0.5,
    initialBackground: 0.1,
    initialBranchingRatio: 0.4,
    initialDecayRate: 0.2,
    initialSpatialStandardDeviation: 0.05
  });
  expect(Array.from(values)).toEqual([4, 365, 30, 0.5, 1e-4, 0.1, 0.4, 0.2, 0.05].map(Math.fround));
  const settings = {
    windowArea: 4,
    windowDuration: 365,
    maximumLag: 30,
    maximumDistance: 0.5,
    initialBackground: 0.1,
    initialBranchingRatio: 0.4,
    initialDecayRate: 0.2,
    initialSpatialStandardDeviation: 0.05
  };
  for (const name of Object.keys(settings) as (keyof typeof settings)[]) {
    expect(() => getGPUSpaceTimeHawkesParameterValues({...settings, [name]: 0})).toThrow(
      `getGPUSpaceTimeHawkesParameterValues ${name} must be positive and finite`
    );
    expect(() => getGPUSpaceTimeHawkesParameterValues({...settings, [name]: Number.NaN})).toThrow(
      `getGPUSpaceTimeHawkesParameterValues ${name} must be positive and finite`
    );
  }
  expect(() => getGPUSpaceTimeHawkesParameterValues({...settings, tolerance: -1})).toThrow(
    'tolerance must be positive and finite'
  );
  expect(() => getGPUSpaceTimeHawkesParameterValues(settings, new Float32Array(3))).toThrow(
    'target must hold 9 elements'
  );
});

it('the binary64 reference computes the E-step and M-step of a three event catalogue by hand', () => {
  // Hand calculation, independent of the reference code. Window: area 2, duration 10. Events at
  // (0,0) t=1, (1,0) t=2, (0,0) t=3. Parameters mu 0.1, omega 0.5, beta 1, sigma 1 (variance 1).
  // c = omega beta / (2 pi) = 0.5 / (2 pi) = 0.0795774715.
  // Event 2: parent 1, lag 1, d2 = 1: term = c exp(-1) exp(-1/2) = c exp(-1.5) = 0.0177558.
  //   lambda = 0.1177558; p_background = 0.1 / lambda = 0.849215; p_parent = 0.150785.
  // Event 3: parent 1 (lag 2, d2 = 0): c exp(-2) = 0.0107698; parent 2 (lag 1, d2 = 1): 0.0177558.
  //   lambda = 0.1285256; p_background = 0.778059; p_1 = 0.083796; p_2 = 0.138145.
  // Event 1 has no parent: p_background = 1.
  const c = 0.5 / (2 * Math.PI);
  const term21 = c * Math.exp(-1.5);
  const term31 = c * Math.exp(-2);
  const term32 = c * Math.exp(-1.5);
  const lambda2 = 0.1 + term21;
  const lambda3 = 0.1 + term31 + term32;
  const triggered = term21 / lambda2 + (term31 + term32) / lambda3;
  const lagSum = (term21 * 1) / lambda2 + (term31 * 2 + term32 * 1) / lambda3;
  const distanceSum = (term21 * 1) / lambda2 + (term32 * 1) / lambda3;
  const edge = [1, 2, 3].reduce((sum, time) => sum + (1 - Math.exp(-(10 - time))), 0);
  const backgroundSum = 1 + 0.1 / lambda2 + 0.1 / lambda3;
  const events: HawkesEvent[] = [
    {x: 0, y: 0, time: 1},
    {x: 1, y: 0, time: 2},
    {x: 0, y: 0, time: 3}
  ];
  const window: HawkesWindow = {
    area: 2,
    duration: 10,
    maximumLag: 100,
    maximumDistance: 100,
    maximumParents: 10
  };
  const fit = fitHawkesReference(
    events,
    window,
    {background: 0.1, branchingRatio: 0.5, decayRate: 1, spatialStandardDeviation: 1},
    {maximumIterations: 1, tolerance: 1e-12}
  );
  expect(fit.estimates.background).toBeCloseTo(backgroundSum / (2 * 10), 12);
  expect(fit.estimates.branchingRatio).toBeCloseTo(triggered / edge, 12);
  expect(fit.estimates.decayRate).toBeCloseTo(triggered / lagSum, 12);
  expect(fit.estimates.spatialStandardDeviation).toBeCloseTo(
    Math.sqrt(distanceSum / (2 * triggered)),
    12
  );
  expect(fit.backgroundProbability[0]).toBeGreaterThan(0);
  // Values quoted in the comment above.
  expect(0.1 / lambda2).toBeCloseTo(0.849215, 5);
  expect(0.1 / lambda3).toBeCloseTo(0.778059, 5);
});

it('the binary64 reference recovers known parameters within their sampling variability', () => {
  // The same derivation as the GPU spec, binary64 only: the estimator's mean and standard
  // deviation over independent simulations, and a fresh simulation inside bias plus four SD.
  const truth = {
    background: 2,
    branchingRatio: 0.5,
    decayRate: 2,
    spatialStandardDeviation: 0.02
  };
  const window = {area: 1, duration: 400, maximumLag: 5, maximumDistance: 0.12, maximumParents: 64};
  const initial = {
    background: 1,
    branchingRatio: 0.3,
    decayRate: 1,
    spatialStandardDeviation: 0.05
  };
  const names = ['background', 'branchingRatio', 'decayRate', 'spatialStandardDeviation'] as const;
  const estimates = names.map(() => [] as number[]);
  for (let replicate = 0; replicate < 24; replicate++) {
    const {events} = simulateHawkes(truth, 1, window.duration, 2000 + replicate);
    const fit = fitHawkesReference(events, window, initial, {
      maximumIterations: 100,
      tolerance: 1e-4
    });
    expect(fit.converged).toBe(true);
    names.forEach((name, index) => estimates[index].push(fit.estimates[name]));
  }
  names.forEach((name, index) => {
    const mean = estimates[index].reduce((sum, value) => sum + value, 0) / 24;
    const deviation = Math.sqrt(
      estimates[index].reduce((sum, value) => sum + (value - mean) ** 2, 0) / 23
    );
    // The mean sits within about one standard deviation of the truth: small bias, not drift.
    expect(Math.abs(mean - truth[name]), name).toBeLessThan(deviation + 0.002 * truth[name]);
    // Roughly the Fisher-information scale: a few percent for rates, under one percent for sigma.
    expect(deviation / truth[name], name).toBeLessThan(0.1);
    expect(deviation / truth[name], name).toBeGreaterThan(0.005);
  });
});
