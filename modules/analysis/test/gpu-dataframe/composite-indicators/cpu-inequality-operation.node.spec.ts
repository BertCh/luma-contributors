// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  CPUInequalityOperation,
  computeInequalityOnCPU
} from '../../../src/dataframe-analysis/composite-indicators/cpu-inequality-operation';
import {GPUInequalityOperation} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality-operation';
import {getGPUInequalityParameterValues} from '../../../src/dataframe-analysis/composite-indicators/inequality-parameters';

it('delegates the GPU-only knot cap to contributor construction at lowering', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
      createTransientView(
        graph,
        port,
        port === 'zoneIds' ? 'uint32' : 'float32',
        port === 'parameters' ? 4 : 2
      ) as GraphDataView<Format>;
    const operation = new GPUInequalityOperation({
      id: 'gpu-inequality',
      values: 'values',
      zoneIds: 'zoneIds',
      parameters: 'parameters',
      zoneCount: 2,
      lorenzKnotCount: 1026,
      output: {gini: 'gini'}
    });
    expect(() => operation.createContributor(getView)).toThrow(/gpu-inequality lorenzKnotCount/);
  } finally {
    device.destroy();
  }
});

it('matches the hand-computed two-income distribution and every summary field', () => {
  const result = computeInequalityOnCPU({
    values: Float32Array.of(1, 3),
    zoneIds: Uint32Array.of(0, 0),
    zoneCount: 2,
    parameters: getGPUInequalityParameterValues({palmaTopShare: 0.5, palmaBottomShare: 0.5}),
    lorenzKnotCount: 5
  });
  expect(result.gini[0]).toBe(0.25);
  expect(result.theilT[0]).toBeCloseTo(Math.log(27 / 16) / 4, 6);
  expect(result.theilL[0]).toBeCloseTo(Math.log(4 / 3) / 2, 6);
  expect(result.atkinson[0]).toBeCloseTo(1 - Math.sqrt(3) / 2, 6);
  expect(result.hoover[0]).toBe(0.25);
  expect(result.palma[0]).toBe(3);
  expect(result.mean[0]).toBe(2);
  expect(Array.from(result.counts)).toEqual([2, 0]);
  expect(Array.from(result.lorenzKnots.slice(0, 5))).toEqual([0, 0.125, 0.25, 0.625, 1]);
  for (const name of ['gini', 'theilT', 'theilL', 'atkinson', 'hoover', 'palma', 'mean'] as const)
    expect(result[name][1]).toBeNaN();
  expect(result.lorenzKnots.slice(5).every(Number.isNaN)).toBe(true);
  expect(Array.from(result.globalSummary)).toEqual([
    result.theilT[0],
    0,
    result.theilT[0],
    0.25,
    2,
    2,
    2,
    4
  ]);
});

it('defines the empty-input extension by zero counts and totals and undefined statistics', () => {
  const result = computeInequalityOnCPU({
    values: new Float32Array(),
    zoneIds: new Uint32Array(),
    zoneCount: 1,
    parameters: getGPUInequalityParameterValues()
  });
  expect(Array.from(result.counts)).toEqual([0]);
  for (const name of [
    'gini',
    'theilT',
    'theilL',
    'atkinson',
    'hoover',
    'palma',
    'mean',
    'lorenzKnots'
  ] as const)
    expect(result[name].every(Number.isNaN)).toBe(true);
  // No rows contribute any population or income; every normalized statistic is undefined.
  expect(Array.from(result.globalSummary)).toEqual([NaN, NaN, NaN, NaN, 0, NaN, 0, 0]);
});

it('matches an independent pairwise weighted Gini and preserves replication and decomposition invariants', () => {
  const values = Float32Array.of(1, 3, 7, 0);
  const weights = Float32Array.of(2, 1, 3, 1);
  const input = {
    values,
    weights,
    zoneIds: Uint32Array.of(0, 0, 1, 1),
    zoneCount: 2,
    parameters: getGPUInequalityParameterValues({epsilon: 0.5})
  };
  const result = computeInequalityOnCPU(input);
  let pairwiseDifference = 0;
  for (let first = 0; first < values.length; first++) {
    for (let second = 0; second < values.length; second++)
      pairwiseDifference +=
        weights[first] * weights[second] * Math.abs(values[first] - values[second]);
  }
  // W=7, S=26; pairwise definition uses all ordered pairs, not Lorenz trapezoids.
  expect(result.globalSummary[3]).toBeCloseTo(pairwiseDifference / (2 * 7 * 26), 6);
  expect(result.globalSummary[0]).toBeCloseTo(result.globalSummary[1] + result.globalSummary[2], 6);
  const repeated = computeInequalityOnCPU({
    ...input,
    values: Float32Array.of(1, 1, 3, 7, 7, 7, 0),
    zoneIds: Uint32Array.of(0, 0, 0, 1, 1, 1, 1),
    weights: undefined
  });
  for (const name of [
    'gini',
    'theilT',
    'theilL',
    'atkinson',
    'hoover',
    'palma',
    'mean',
    'lorenzKnots'
  ] as const)
    expect(Array.from(result[name])).toEqual(Array.from(repeated[name]));
});

it('handles zero income, filtering, alternate epsilon and mismatched input arrays', () => {
  const input = {
    values: Float32Array.of(0, 0, -1, NaN, 9, 9, 9),
    zoneIds: Uint32Array.of(0, 0, 0, 0, 0, 0xffffffff, 0),
    mask: Uint32Array.of(1, 1, 1, 1, 0, 1, 1),
    weights: Float32Array.of(1, 1, 1, 1, 1, 1, 0),
    zoneCount: 1,
    parameters: getGPUInequalityParameterValues()
  };
  const result = computeInequalityOnCPU(input);
  expect(result.counts[0]).toBe(2);
  expect(result.mean[0]).toBe(0);
  expect(result.gini[0]).toBeNaN();
  expect(result.lorenzKnots.every(Number.isNaN)).toBe(true);
  expect(Array.from(result.globalSummary)).toEqual([NaN, NaN, NaN, NaN, 2, 0, 2, 0]);
  for (const [epsilon, expected] of [
    [0, 0],
    [0.5, (2 - Math.sqrt(3)) / 4],
    [2, 0.25]
  ]) {
    expect(
      computeInequalityOnCPU({
        values: Float32Array.of(1, 3),
        zoneIds: Uint32Array.of(0, 0),
        zoneCount: 1,
        parameters: getGPUInequalityParameterValues({epsilon})
      }).atkinson[0]
    ).toBeCloseTo(expected, 6);
  }
  for (const name of ['zoneIds', 'weights', 'mask'] as const)
    expect(() =>
      computeInequalityOnCPU({...input, [name]: new Uint32Array(1)} as typeof input)
    ).toThrow(name);
  expect(() => computeInequalityOnCPU({...input, parameters: new Float32Array(1)})).toThrow(
    'parameters'
  );
});

it('validates operation shapes and keeps both graph output contracts identical', () => {
  const props = {
    id: 'inequality-test',
    values: 'values',
    zoneIds: 'zoneIds',
    parameters: 'parameters',
    zoneCount: 2,
    output: {gini: 'gini', counts: 'counts', globalSummary: 'summary', lorenzKnots: 'knots'}
  };
  expect(() => new CPUInequalityOperation({...props, zoneCount: 0})).toThrow(
    /inequality-test zoneCount/
  );
  expect(() => new CPUInequalityOperation({...props, lorenzKnotCount: 1})).toThrow(
    /inequality-test lorenzKnotCount/
  );
  expect(() => new CPUInequalityOperation({...props, output: {}})).toThrow(
    /inequality-test output/
  );
  expect(() => new GPUInequalityOperation({...props, zoneCount: 0})).toThrow(
    /inequality-test outputs/
  );
  expect(new CPUInequalityOperation(props).contract.outputs).toEqual(
    new GPUInequalityOperation(props).contract.outputs
  );
  expect(() => new CPUInequalityOperation({...props, lorenzKnotCount: 1026})).not.toThrow();
});
