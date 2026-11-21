// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {
  GPULocalMoran,
  GPUHotSpotAnalysis,
  GPULocalGeary,
  GPUSpatialPearson,
  GPUGammaStatistic,
  GPUAssuncaoReisStandardization,
  GPUQuantileLISA,
  GPULOSHResidualPermutationTest,
  GPU_LOCAL_GEARY_PERMUTATION_ADAPTER,
  GPU_LOCAL_BIVARIATE_MORAN_PERMUTATION_ADAPTER,
  GPU_LOCAL_JOIN_COUNT_PERMUTATION_ADAPTER,
  GPU_LOCAL_BIVARIATE_JOIN_COUNT_PERMUTATION_ADAPTER,
  GPU_LEES_L_PERMUTATION_ADAPTERS,
  GPU_SPATIAL_PEARSON_PERMUTATION_ADAPTER,
  getGPUGammaPermutationAdapter
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  computeLocalPermutationOracle,
  computeGlobalPermutationOracle
} from '../permutation-inference/permutation-oracle';
import {
  GLOBAL_AUDIT_VALUES,
  GLOBAL_AUDIT_SECOND,
  GLOBAL_AUDIT_MASK,
  GLOBAL_AUDIT_WEIGHTS
} from '../global-spatial-statistics/global-mathematical-audit-fixture';
import {
  HANDOFF_REFERENCE,
  HANDOFF_BINARY_SECOND,
  HANDOFF_GLOBAL_LEE
} from './permutation-handoff-audit-fixture';

it('older contributors capture validated props and nested port maps before caller mutation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `construction-audit-${serial++}`, format, length);
  const weights = () => ({
    offsets: view('uint32', 9),
    neighbors: view('uint32', 8),
    weights: view('float32', 8),
    status: {overflow: view('uint32', 1)}
  });
  const common = () => ({
    weights: weights(),
    values: view('float32', 8),
    parameters: view('float32', 8)
  });
  const check = (contributor: {props: object}, caller: object, key: string) => {
    const captured = contributor.props as {output: Record<string, unknown>};
    const original = caller as {output: Record<string, unknown>};
    const expected = original.output[key];
    Reflect.set(original.output, key, view('float32', 8));
    expect(captured.output[key], `${contributor.constructor.name} retains validated output`).toBe(
      expected
    );
    expect(captured).not.toBe(original);
    expect(Object.isFrozen(captured)).toBe(true);
    expect(Object.isFrozen(captured.output)).toBe(true);
  };
  try {
    for (const Contributor of [GPULocalMoran, GPUHotSpotAnalysis]) {
      const props = {...common(), output: {zScores: view('float32', 8)}};
      check(new Contributor(props), props, 'zScores');
    }
    for (const Contributor of [GPULocalGeary, GPUSpatialPearson, GPUGammaStatistic]) {
      const props = {
        ...common(),
        secondValues: view('float32', 8),
        output: {statistic: view('float32', Contributor === GPULocalGeary ? 8 : 1)}
      };
      check(new Contributor(props), props, 'statistic');
    }
    expect
      .soft(
        () =>
          new GPUSpatialPearson({
            ...common(),
            secondValues: undefined!,
            output: {statistic: view('float32', 1)}
          })
      )
      .toThrow(/secondValues/);
    const rate = {
      events: view('float32', 8),
      populations: view('float32', 8),
      output: {standardizedRates: view('float32', 8)}
    };
    check(new GPUAssuncaoReisStandardization(rate), rate, 'standardizedRates');
    const global = {
      ...common(),
      statistics: ['moran' as const],
      output: {summary: view('float32', 52)}
    };
    const globalContributor = new GPUGlobalSpatialStatistics(global);
    check(globalContributor, global, 'summary');
    global.statistics[0] = 'geary' as 'moran';
    expect(globalContributor.props.statistics).toEqual(['moran']);
    const expectedOffsets = global.weights.offsets;
    global.weights.offsets = view('uint32', 2);
    expect(
      'offsets' in globalContributor.props.weights && globalContributor.props.weights.offsets
    ).toBe(expectedOffsets);
    const expectedOverflow = global.weights.status.overflow;
    global.weights.status.overflow = view('uint32', 1);
    expect(globalContributor.props.weights.status?.overflow).toBe(expectedOverflow);
    const quantile = {
      alternative: 'geoda-legacy' as const,
      ...common(),
      parameters: view('uint32', 4),
      quantileParameters: view('uint32', 2),
      maximumPermutations: 9,
      additionalVariables: [{values: view('float32', 8), quantileParameters: view('uint32', 2)}],
      output: {
        membership: view('uint32', 8),
        boundaries: view('float32', 4),
        validCount: view('uint32', 2),
        classes: view('uint32', 8),
        invalidQuantile: view('uint32', 1),
        exceedances: view('uint32', 8),
        pseudoPValues: view('float32', 8),
        status: {overflow: view('uint32', 1)}
      }
    };
    const quantileContributor = new GPUQuantileLISA(quantile);
    check(quantileContributor, quantile, 'membership');
    const expectedValues = quantile.additionalVariables[0].values;
    quantile.additionalVariables[0].values = view('float32', 8);
    expect(quantileContributor.props.additionalVariables?.[0].values).toBe(expectedValues);
    const losh = {
      ...common(),
      parameters: view('uint32', 4),
      maximumPermutations: 9,
      alternative: 'greater' as const,
      output: {
        statistic: view('float32', 8),
        status: view('uint32', 8),
        inference: {
          exceedances: view('uint32', 8),
          pseudoPValues: view('float32', 8),
          status: {
            overflow: view('uint32', 1),
            weightsOverflow: view('uint32', 1),
            neighborOverflow: view('uint32', 1),
            invalidInput: view('uint32', 1)
          }
        }
      }
    };
    const loshContributor = new GPULOSHResidualPermutationTest(losh);
    expect(() => new GPULOSHResidualPermutationTest({...losh, maximumPermutations: 0})).toThrow(
      /maximumPermutations/
    );
    const loshNodes = loshContributor.getCommandNodes(graph);
    for (const name of ['weightsOverflow', 'neighborOverflow', 'invalidInput'] as const) {
      const destination = losh.output.inference.status[name];
      expect(loshContributor.props.output.inference.status?.[name]).toBe(destination);
      expect(
        loshNodes.some(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              resource.buffer === destination &&
              resource.usage === 'storage-read-write'
          )
        )
      ).toBe(true);
    }
    check(loshContributor, losh, 'statistic');
    expect(loshContributor.props.output.status).toBe(losh.output.status);
    const expectedInference = losh.output.inference.pseudoPValues;
    losh.output.inference.pseudoPValues = view('float32', 8);
    expect(loshContributor.props.output.inference.pseudoPValues).toBe(expectedInference);
  } finally {
    device.destroy();
  }
});

it('public adapters select the documented statistic, donor variable, scope and tail', () => {
  expect(GPU_LOCAL_GEARY_PERMUTATION_ADAPTER).toEqual({
    scope: 'local',
    statistic: 'localGeary',
    permutedVariable: 'values',
    alternative: 'lesser'
  });
  expect(GPU_LOCAL_BIVARIATE_MORAN_PERMUTATION_ADAPTER).toEqual({
    scope: 'local',
    statistic: 'localBivariateMoran',
    permutedVariable: 'secondValues',
    alternative: 'two-sided'
  });
  expect(GPU_LOCAL_JOIN_COUNT_PERMUTATION_ADAPTER).toEqual({
    scope: 'local',
    statistic: 'localJoinCount',
    permutedVariable: 'values',
    alternative: 'greater'
  });
  expect(GPU_LOCAL_BIVARIATE_JOIN_COUNT_PERMUTATION_ADAPTER).toEqual({
    scope: 'local',
    statistic: 'localBivariateJoinCount',
    permutedVariable: 'secondValues',
    alternative: 'greater'
  });
  expect(GPU_LEES_L_PERMUTATION_ADAPTERS).toEqual({
    local: {
      scope: 'local',
      statistic: 'localLeeL',
      permutedVariable: 'values',
      alternative: 'two-sided'
    },
    global: {
      scope: 'global',
      statistic: 'leesL',
      permutedVariable: 'values',
      alternative: 'two-sided'
    }
  });
  expect(GPU_SPATIAL_PEARSON_PERMUTATION_ADAPTER).toEqual({
    scope: 'global',
    statistic: 'bivariateMoran',
    permutedVariable: 'secondValues',
    alternative: 'two-sided'
  });
  expect(getGPUGammaPermutationAdapter('cross-product')).toEqual({
    scope: 'global',
    statistic: 'moran',
    permutedVariable: 'values',
    alternative: 'two-sided'
  });
  expect(getGPUGammaPermutationAdapter('squared-difference')).toEqual({
    scope: 'global',
    statistic: 'geary',
    permutedVariable: 'values',
    alternative: 'two-sided'
  });
  expect(() => getGPUGammaPermutationAdapter('absolute-difference')).toThrow(/absolute-difference/);
  expect(() => getGPUGammaPermutationAdapter('unknown' as 'cross-product')).toThrow(/operation/);
});

it('local bivariate, Lee and join-count observed handoffs match independent dense definitions', () => {
  const weights = {
    offsets: Uint32Array.from(GLOBAL_AUDIT_WEIGHTS.offsets),
    neighbors: Uint32Array.from(GLOBAL_AUDIT_WEIGHTS.neighbors),
    weights: Float32Array.from(GLOBAL_AUDIT_WEIGHTS.weights)
  };
  const maximumDifferences: Record<string, number> = {};
  for (const name of [
    'localBivariateMoran',
    'localLeeL',
    'localJoinCount',
    'BJC',
    'CLC'
  ] as const) {
    const binary = ['localJoinCount', 'BJC', 'CLC'].includes(name);
    const values = Float32Array.from(GLOBAL_AUDIT_VALUES, value =>
      binary && Number.isFinite(value) ? Number(value !== 0) : value
    );
    const result = computeLocalPermutationOracle({
      weights,
      values,
      secondValues:
        name === 'localJoinCount'
          ? undefined
          : Float32Array.from(binary ? HANDOFF_BINARY_SECOND : GLOBAL_AUDIT_SECOND),
      mask: Uint32Array.from(GLOBAL_AUDIT_MASK),
      statistic: name === 'BJC' || name === 'CLC' ? 'localBivariateJoinCount' : name,
      joinCase: name === 'BJC' || name === 'CLC' ? name : undefined,
      seed: 43,
      permutationCount: 19,
      significanceLevel: 0.6,
      maximumNeighbors: 4
    });
    for (const [row, expected] of HANDOFF_REFERENCE[name].entries()) {
      if (Number.isNaN(expected)) expect(result.observed[row]).toBeNaN();
      else {
        expect(Number.isFinite(result.observed[row])).toBe(true);
        const difference = Math.abs(result.observed[row] - expected);
        maximumDifferences[name] = Math.max(maximumDifferences[name] ?? 0, difference);
        expect(difference).toBeLessThan(1e-5);
      }
    }
  }
  const global = computeGlobalPermutationOracle({
    weights,
    values: Float32Array.from(GLOBAL_AUDIT_VALUES),
    secondValues: Float32Array.from(GLOBAL_AUDIT_SECOND),
    mask: Uint32Array.from(GLOBAL_AUDIT_MASK),
    statistic: 'leesL',
    seed: 43,
    permutationCount: 19
  });
  expect(Number.isFinite(global.observed)).toBe(true);
  expect(Math.abs(global.observed - HANDOFF_GLOBAL_LEE)).toBeLessThan(1e-14);
  console.info(
    'independent handoff maximum observed errors',
    maximumDifferences,
    'global Lee',
    Math.abs(global.observed - HANDOFF_GLOBAL_LEE)
  );
});
