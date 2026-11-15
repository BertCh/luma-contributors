// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUVariogramModelFitOperation,
  computeVariogramModelFitOnCPU
} from '../../../src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation';
import {GPUVariogramModelFitOperation} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit-operation';
import type {VariogramModelType} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';

const LAGS = Array.from({length: 24}, (_, index) => (index + 1) * 0.25);
const PAIR_COUNTS = LAGS.map((_, index) => 50 + 3 * index);

// The model families written out from their definitions (effective range: the spherical model
// reaches the sill there, the others reach 95% of it), independent of the source's shape function.
const SHAPES: Record<'spherical' | 'exponential' | 'gaussian', (scaled: number) => number> = {
  spherical: scaled => (scaled >= 1 ? 1 : 1.5 * scaled - 0.5 * scaled ** 3),
  exponential: scaled => 1 - Math.exp(-3 * scaled),
  gaussian: scaled => 1 - Math.exp(-3 * scaled * scaled)
};

function getBins(model: keyof typeof SHAPES, nugget: number, sill: number, range: number) {
  return {
    distances: LAGS,
    semivariances: LAGS.map(lag => nugget + sill * SHAPES[model](lag / range)),
    pairCounts: PAIR_COUNTS
  };
}

for (const weighting of ['pairs', 'none', 'cressie'] as const) {
  it(`the CPU fit with ${weighting} weights recovers the parameters exact bins were drawn from`, () => {
    const truths = {
      spherical: {nugget: 0.3, sill: 2, range: 3.1},
      exponential: {nugget: 0.1, sill: 1.2, range: 2.4},
      gaussian: {nugget: 0.5, sill: 0.8, range: 1.7}
    };
    for (const [model, truth] of Object.entries(truths) as [
      keyof typeof SHAPES,
      (typeof truths)['spherical']
    ][]) {
      const result = computeVariogramModelFitOnCPU(
        getBins(model, truth.nugget, truth.sill, truth.range),
        {models: [model], weighting}
      );
      const [nugget, sill, range, residual] = result.parameters;
      // Exact bins: any weighting has its minimum, zero, at the truth.
      expect(nugget).toBeCloseTo(truth.nugget, 4);
      expect(sill).toBeCloseTo(truth.sill, 4);
      expect(range).toBeCloseTo(truth.range, 4);
      expect(residual).toBeLessThan(1e-9);
      expect(result.status[0]).toBe(0);
      const [binCount, unweightedResidual, informationCriterion, bracket] = result.diagnostics;
      expect(binCount).toBe(LAGS.length);
      expect(unweightedResidual).toBeLessThan(1e-9);
      // AIC of a three-parameter fit: n * ln(max(RSS / n, 1e-30)) + 6, rounded to float32.
      expect(informationCriterion).toBe(
        Math.fround(
          LAGS.length * Math.log(Math.max(result.diagnostics[1] / LAGS.length, 1e-30)) + 6
        )
      );
      expect(bracket).toBeGreaterThanOrEqual(0);
      expect(bracket).toBeLessThan(1e-9);
    }
  });
}

it('the CPU fit writes one row per model in the order of models', () => {
  const models: VariogramModelType[] = ['gaussian', 'exponential', 'spherical'];
  const result = computeVariogramModelFitOnCPU(getBins('exponential', 0.2, 1.5, 2.2), {
    models,
    weighting: 'pairs'
  });
  expect(result.parameters.length).toBe(12);
  expect(result.diagnostics.length).toBe(12);
  expect(result.status.length).toBe(3);
  // Only the family the bins came from reproduces them.
  const residuals = models.map((_, row) => result.parameters[row * 4 + 3]);
  expect(residuals[1]).toBeLessThan(1e-9);
  expect(residuals[0]).toBeGreaterThan(1e-3);
  expect(residuals[2]).toBeGreaterThan(1e-3);
});

it('the CPU fit reports status 1 and NaN rows without two usable bins at two lags', () => {
  const cases = [
    // No pairs anywhere.
    {distances: [1, 2, 3], semivariances: [1, 2, 3], pairCounts: [0, 0, 0]},
    // One usable bin: the others have a non-finite value, a non-positive lag or no pairs.
    {distances: [1, 2, 0, 4], semivariances: [1, Number.NaN, 3, 4], pairCounts: [5, 5, 5, 0]},
    // Two usable bins at the same lag.
    {distances: [2, 2], semivariances: [1, 1.5], pairCounts: [5, 5]}
  ];
  for (const bins of cases) {
    const result = computeVariogramModelFitOnCPU(bins, {models: ['spherical', 'gaussian']});
    expect(Array.from(result.status)).toEqual([1, 1]);
    expect(result.parameters.every(Number.isNaN)).toBe(true);
    expect(result.diagnostics.every(Number.isNaN)).toBe(true);
  }
});

it('the CPU fit reports status 3 when the range runs to the end of the search domain', () => {
  // Semivariance still rising linearly at the largest lag: no sill inside three times that lag.
  const bins = {
    distances: LAGS,
    semivariances: LAGS.map(lag => 0.5 * lag),
    pairCounts: PAIR_COUNTS
  };
  const result = computeVariogramModelFitOnCPU(bins, {models: ['spherical'], weighting: 'none'});
  expect(result.status[0]).toBe(3);
  expect(result.parameters[2]).toBeCloseTo(3 * LAGS[LAGS.length - 1], 3);
});

it('the CPU fit reports status 2 when a fitted value does not fit float32', () => {
  const result = computeVariogramModelFitOnCPU(
    {distances: LAGS, semivariances: LAGS.map(lag => 1e39 * lag), pairCounts: PAIR_COUNTS},
    {models: ['exponential'], weighting: 'none'}
  );
  expect(result.status[0]).toBe(2);
  expect(result.parameters.every(Number.isNaN)).toBe(true);
});

it('both operations reject unknown models and weightings at construction, naming the ID', () => {
  const props = {
    id: 'fit',
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairCounts',
    output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
  };
  expect(() => new CPUVariogramModelFitOperation({...props, models: []})).toThrow(
    'fit models must name one to five model families'
  );
  expect(
    () => new CPUVariogramModelFitOperation({...props, weighting: 'inverse' as 'pairs'})
  ).toThrow("fit weighting must be 'cressie', 'pairs' or 'none'");
  expect(
    () => new CPUVariogramModelFitOperation({...props, models: ['cubic' as 'spherical']})
  ).toThrow();
  // The two classes take the same properties and declare the same graph values.
  const cpu = new CPUVariogramModelFitOperation(props);
  const gpu = new GPUVariogramModelFitOperation(props);
  expect(cpu.id).toBe(gpu.id);
  const getNames = (contract: typeof cpu.contract) =>
    contract.values.map(use => `${use.access}:${String(use.value)}`).sort();
  expect(getNames(cpu.contract)).toEqual(getNames(gpu.contract));
  expect(cpu.contract.outputs).toEqual(gpu.contract.outputs);
});

it.each([
  'distances',
  'semivariances',
  'pairCounts'
] as const)('both operations reject invalid graph names for input %s at construction', port => {
  for (const value of ['', undefined, null, 0, false, {id: 'bins'}, ['bins']]) {
    const props = {
      id: 'fit',
      distances: 'distances',
      semivariances: 'semivariances',
      pairCounts: 'pairCounts',
      [port]: value as string,
      output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
    };
    for (const operationClass of [GPUVariogramModelFitOperation, CPUVariogramModelFitOperation]) {
      expect(() => new operationClass(props)).toThrow(
        `fit inputs.${port} must be a non-empty graph value name`
      );
    }
  }
});

it.each([
  'parameters',
  'diagnostics',
  'status'
] as const)('both operations reject invalid graph names for output %s before names become object keys', port => {
  for (const value of ['', undefined, null, 0, false, {id: 'fitted'}, ['fitted']]) {
    const props = {
      id: 'fit',
      distances: 'distances',
      semivariances: 'semivariances',
      pairCounts: 'pairCounts',
      output: {
        parameters: 'parameters',
        diagnostics: 'diagnostics',
        status: 'status',
        [port]: value as string
      }
    };
    for (const operationClass of [GPUVariogramModelFitOperation, CPUVariogramModelFitOperation]) {
      expect(() => new operationClass(props)).toThrow(
        `fit outputs.${port}.value must be a non-empty graph value name`
      );
    }
  }
});
