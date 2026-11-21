// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeInequalityOnCPU,
  CPUInequalityOperation
} from '../../../src/dataframe-analysis/composite-indicators/cpu-inequality-operation';
import {
  getGPUInequalityParameterValues,
  GPU_INEQUALITY_GLOBAL_SUMMARY,
  GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH
} from '../../../src/dataframe-analysis/composite-indicators/inequality-parameters';
import {computeInequalityOnCPU as computeOracle} from './inequality-oracle';

it.each([
  {values: [0, 1], weights: [1, 2 ** 58], expected: 3.469446951953614e-18},
  {values: [0, 1], weights: [2 ** 58, 1], expected: 1},
  {values: [1, 2], weights: [2 ** 58, 1], expected: 3.469446951953614e-18},
  {values: [1, 1 + 2 ** -23], weights: [1, 1], expected: 2.9802320611338473e-8}
])('preserves weighted Gini with a small endpoint weight: %j', fixture => {
  const result = computeInequalityOnCPU({
    values: Float32Array.from(fixture.values),
    weights: Float32Array.from(fixture.weights),
    zoneIds: Uint32Array.of(0, 0),
    zoneCount: 1,
    parameters: getGPUInequalityParameterValues()
  });
  // Pair-difference definition: w0*w1*|x0-x1| / ((w0+w1)*(w0*x0+w1*x1)).
  // Exact fractions are 1/(2^58+1), 2^58/(2^58+1), and
  // 2^58/((2^58+1)*(2^58+2)); literals above are their float32 roundings.
  // The unit-weight adjacent-float fixture is (2^-23)/(4 + 2^-22).
  expect(result.weightedGini).toHaveLength(1);
  expect(result.globalSummary).toHaveLength(GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH);
  expect(Number.isFinite(result.weightedGini[0])).toBe(true);
  expect(result.weightedGini[0]).toBe(fixture.expected);
  expect(result.globalSummary[GPU_INEQUALITY_GLOBAL_SUMMARY.GINI]).toBe(fixture.expected);
  const oracle = computeOracle({...fixture, zoneIds: [0, 0], zoneCount: 1});
  expect(Math.fround(oracle.gini[0])).toBe(fixture.expected);
  expect(Math.fround(oracle.globalSummary[GPU_INEQUALITY_GLOBAL_SUMMARY.GINI])).toBe(
    fixture.expected
  );
});

it('returns exact zero Gini for constant positive incomes in zones and the pooled population', () => {
  const result = computeInequalityOnCPU({
    values: new Float32Array(24).fill(1),
    zoneIds: Uint32Array.from({length: 24}, (_, row) => row % 2),
    zoneCount: 2,
    parameters: getGPUInequalityParameterValues()
  });
  // Every pairwise income difference is exactly zero, regardless of the population size.
  expect(Array.from(result.weightedGini)).toEqual([0, 0]);
  expect(result.globalSummary[GPU_INEQUALITY_GLOBAL_SUMMARY.GINI]).toBe(0);
});

it('exposes the weighted Gini in the CPU result', () => {
  const result = computeInequalityOnCPU({
    values: Float32Array.of(1, 3),
    zoneIds: Uint32Array.of(0, 0),
    zoneCount: 1,
    parameters: getGPUInequalityParameterValues()
  });
  expect(result).not.toHaveProperty('gini');
  expect(result.weightedGini[0]).toBe(0.25);
});

const operationProps = {
  id: 'inequality-names',
  values: 'income',
  zoneIds: 'zones',
  parameters: 'settings',
  weights: 'population',
  mask: 'included',
  zoneCount: 1
};

it.each([
  {gini: ''},
  {gini: 'result', counts: 'result'},
  {lorenzKnots: 'result', globalSummary: 'result'},
  {globalSummary: ''},
  {counts: 'income'},
  {lorenzKnots: 'zones'},
  {globalSummary: 'settings'},
  {mean: 'population'},
  {palma: 'included'}
])('rejects invalid CPU destinations before keying declarations: %j', output => {
  expect(() => new CPUInequalityOperation({...operationProps, output})).toThrow(
    /^inequality-names outputs\./
  );
});

it('accepts a minimal optional output and ignores unselected outputs', () => {
  const operation = new CPUInequalityOperation({
    ...operationProps,
    output: {gini: undefined, globalSummary: 'summary'}
  });
  const outputs = operation.contract.outputs;
  expect(outputs).toBeDefined();
  if (!outputs || typeof outputs === 'function') throw new Error('Expected static outputs');
  expect(Object.keys(outputs)).toEqual(['summary']);
});

it.each([
  'weightedGini',
  'unrecognizedOutput'
])('rejects an unsupported CPU output port: %s', name => {
  // A selection variable can have extra structural fields even in a TypeScript caller.
  const output = {mean: 'm', [name]: 'extra'};
  expect(() => new CPUInequalityOperation({...operationProps, output})).toThrow(
    name === 'weightedGini'
      ? /inequality-names output\.weightedGini.*host.*output\.gini/
      : /inequality-names output\.unrecognizedOutput.*supported/
  );
});

it.each(['weights', 'mask'] as const)('rejects a supplied empty CPU %s name', name => {
  expect(
    () => new CPUInequalityOperation({...operationProps, [name]: '', output: {mean: 'm'}})
  ).toThrow(new RegExp(`inequality-names inputs\\.${name}`));
});
