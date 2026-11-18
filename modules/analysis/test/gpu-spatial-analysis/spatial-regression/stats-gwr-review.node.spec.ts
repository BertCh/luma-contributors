// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUGeographicallyWeightedRegressionParameterValues,
  decodeGPUGeographicallyWeightedRegressionStatus,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  GPUMultiscaleGeographicallyWeightedRegression,
  decodeGPUMultiscaleGeographicallyWeightedRegressionStatus,
  GPU_MULTISCALE_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {computeMultiscaleGeographicallyWeightedRegressionOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPURestrictedRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {getRegressionTileLayout} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-row-tiles';

it('bounds tiled-panel serial merges at 4096 partials with the default tile layout', () => {
  expect(getRegressionTileLayout(64 * 4096)).toEqual({tileRowCount: 64, tileCount: 4096});
  expect(getRegressionTileLayout(64 * 4096 + 1)).toEqual({tileRowCount: 65, tileCount: 4033});
  expect(getRegressionTileLayout(2 ** 24)).toEqual({tileRowCount: 4096, tileCount: 4096});
});

it('enforces the coefficient and entity bounds used by the serial-loop review', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `loop-bound-${serial++}`, format, length);
  const restricted = (predictorCount: number, regimeCount: number) => {
    const coefficientCount = (predictorCount + 1) * regimeCount;
    return new GPURestrictedRegimeRegression({
      predictorCount,
      regimeCount,
      common: new Array(predictorCount + 1).fill(false),
      predictors: view('float32', 64 * predictorCount),
      response: view('float32', 64),
      regimes: view('uint32', 64),
      output: {
        coefficients: view('float32', coefficientCount),
        covariance: view('float32', coefficientCount ** 2),
        fitted: view('float32', 64),
        chow: view('float32', 3),
        status: view('uint32', 1)
      }
    });
  };
  const panel = (entityCount: number) => {
    const rowCount = entityCount * 2;
    return new GPUSpatialPanelLagMaximumLikelihood({
      fitMode: 'tiled',
      effects: 'unit',
      predictorCount: 1,
      periodCount: 2,
      eigenvalues: view('float32', entityCount),
      predictors: view('float32', rowCount),
      response: view('float32', rowCount),
      weights: {
        offsets: view('uint32', entityCount + 1),
        neighbors: view('uint32', entityCount),
        weights: view('float32', entityCount)
      },
      output: {
        coefficients: view('float32', 4),
        covariance: view('float32', 9),
        summary: view('float32', 6),
        status: view('uint32', 1),
        fitted: view('float32', rowCount),
        residuals: view('float32', rowCount)
      }
    });
  };
  try {
    expect(restricted(1, 8).coefficientCount).toBe(16);
    expect(() => restricted(2, 6)).toThrow(/unrestricted coefficients.*16/);
    expect(panel(1024).fitMode).toBe('tiled');
    expect(() => panel(1025)).toThrow(/entities.*1024/);
  } finally {
    device.destroy();
  }
});

it('decodes model-specific status words without confusing singular and excluded', () => {
  expect(decodeGPUGeographicallyWeightedRegressionStatus(1)).toBe('singular');
  expect(decodeGPUMultiscaleGeographicallyWeightedRegressionStatus(1)).toBe('excluded');
  expect(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS).toEqual({OK: 0, SINGULAR: 1, EXCLUDED: 2});
  expect(GPU_MULTISCALE_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS).toEqual({
    CONVERGED: 0,
    EXCLUDED: 1,
    SINGULAR: 2,
    ITERATION_LIMIT: 3,
    NUMERICAL_FAILURE: 4
  });
  expect([0, 1, 2, 3, NaN].map(decodeGPUGeographicallyWeightedRegressionStatus)).toEqual([
    'ok',
    'singular',
    'excluded',
    'unknown',
    'unknown'
  ]);
  expect(
    [0, 1, 2, 3, 4, 5, NaN].map(decodeGPUMultiscaleGeographicallyWeightedRegressionStatus)
  ).toEqual([
    'converged',
    'excluded',
    'singular',
    'iteration-limit',
    'numerical-failure',
    'unknown',
    'unknown'
  ]);
});

for (const significanceLevel of [1 - 2 ** -27, Number.MIN_VALUE]) {
  it(`rejects significanceLevel ${significanceLevel} that packs outside (0, 1)`, () => {
    expect(() =>
      getGPUGeographicallyWeightedRegressionParameterValues({bandwidths: [1], significanceLevel})
    ).toThrow(/significanceLevel/);
  });
}

for (const settings of [
  {bandwidths: [1, 1e100]},
  {bandwidths: [1, NaN]},
  {bandwidths: [1], significanceLevel: 1 - 2 ** -27}
]) {
  it(`rejects unrepresentable settings without modifying the target: ${String(settings.bandwidths)}/${settings.significanceLevel}`, () => {
    const target = new Float32Array(8).fill(123);
    expect(() =>
      getGPUGeographicallyWeightedRegressionParameterValues(settings, 2, target)
    ).toThrow();
    expect(Array.from(target)).toEqual(new Array(8).fill(123));
  });
}

it('reserves the zero significance encoding for omission and preserves finite packed bandwidths', () => {
  expect(getGPUGeographicallyWeightedRegressionParameterValues({bandwidths: [1]}, 1)[3]).toBe(0);
  const result = getGPUGeographicallyWeightedRegressionParameterValues(
    {bandwidths: [1 / 3], significanceLevel: 0.05},
    1
  );
  expect(result[3]).toBe(Math.fround(0.05));
  expect(result[4]).toBe(Math.fround(1 / 3));
});

it('honors one CPU sweep and publishes its finite last fit without converged inference', () => {
  const input = {
    positions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0),
    predictors: Float32Array.of(1, -1, 1, 2),
    response: Float32Array.of(2, 4, 8, 100),
    parameters: Float32Array.of(1e-6, 1, 1),
    mask: Uint32Array.of(1, 1, 1, 0),
    noiseVariance: Float32Array.of(1)
  };
  for (const inference of [undefined, {}]) {
    const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(input, {
      predictorCount: 1,
      kernel: 'bisquare',
      maximumIterations: 1,
      inference
    });
    expect(result.iterationCount[0]).toBe(1);
    expect(Array.from(result.codes)).toEqual([3, 3, 3, 1]);
    // At h=1 only self has positive weight. Initial OLS is [4.5, 0.5], or [0,0]
    // with inference. First intercept update is y-x*slope; slope stays unchanged.
    expect(Array.from(result.coefficients)).toEqual(
      inference ? [2, 0, 4, 0, 8, 0, NaN, NaN] : [1.5, 0.5, 4.5, 0.5, 7.5, 0.5, NaN, NaN]
    );
    expect(Array.from(result.fitted)).toEqual([2, 4, 8, NaN]);
    expect(Array.from(result.conditionalStandardErrors).every(Number.isNaN)).toBe(true);
    if (inference) {
      expect(result.inference!.status[0]).toBe(1);
      expect(Array.from(result.inference!.standardErrors).every(Number.isNaN)).toBe(true);
      expect(Array.from(result.inference!.summary).every(Number.isNaN)).toBe(true);
    }
    const converged = computeMultiscaleGeographicallyWeightedRegressionOnCPU(input, {
      predictorCount: 1,
      kernel: 'bisquare',
      maximumIterations: 2,
      inference
    });
    expect(converged.iterationCount[0]).toBe(2);
    expect(Array.from(converged.codes)).toEqual([0, 0, 0, 1]);
  }
});

it('applies the dense row limit to an explicit grid with an untruncated Gaussian kernel', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32x2' | 'float32' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `stats-gwr-${serial++}`, format, length);
  const construct = (rowCount: number, gaussianCutoff?: number) =>
    new GPUMultiscaleGeographicallyWeightedRegression({
      positions: view('float32x2', rowCount),
      predictors: view('float32', rowCount),
      response: view('float32', rowCount),
      parameters: view('float32', 3),
      predictorCount: 1,
      kernel: 'gaussian',
      maximumIterations: 1,
      indexGridSize: [4, 4],
      gaussianCutoff,
      output: {
        coefficients: view('float32', rowCount * 2),
        fitted: view('float32', rowCount),
        status: {codes: view('uint32', rowCount), iterationCount: view('uint32', 1)}
      }
    });
  try {
    expect(() => construct(65536)).not.toThrow();
    expect(() => construct(65537)).toThrow(/response rows.*65536/);
    expect(() => construct(65537, 5)).not.toThrow();
    expect(() => construct(1048577, 5)).toThrow(/response rows.*1048576/);
  } finally {
    device.destroy();
  }
});
