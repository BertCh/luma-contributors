// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {getEquilibratedSolveWGSL} from '../../../src/gpu-spatial-analysis/spatial-interpolation/equilibrated-solve-wgsl';
import {KRIGING_PIVOT_RATIO_TOLERANCE} from '../../../src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters';
import {
  getGPULocalKrigingParameterLength,
  getGPULocalKrigingParameterValues,
  getPerFramePredictionCovarianceWGSL,
  GPU_LOCAL_KRIGING_PARAMETER_LENGTH,
  MATERN_3_2_RANGE_FACTOR,
  MATERN_5_2_RANGE_FACTOR,
  validatePredictionCovariance
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';

it('Matern range factors put the correlation at exp(-3) at the effective range', () => {
  const at32 = (1 + MATERN_3_2_RANGE_FACTOR) * Math.exp(-MATERN_3_2_RANGE_FACTOR);
  const at52 =
    (1 + MATERN_5_2_RANGE_FACTOR + MATERN_5_2_RANGE_FACTOR ** 2 / 3) *
    Math.exp(-MATERN_5_2_RANGE_FACTOR);
  expect(at32).toBeCloseTo(Math.exp(-3), 12);
  expect(at52).toBeCloseTo(Math.exp(-3), 12);
});

it('packs the documented per-frame layout for one structure', () => {
  const values = getGPULocalKrigingParameterValues({
    covariance: {model: 'matern', smoothness: 1.5, range: 4, sill: 2, nugget: 0.5}
  });
  expect(values.length).toBe(GPU_LOCAL_KRIGING_PARAMETER_LENGTH);
  expect(Array.from(values)).toEqual([2.5, 0, 0, 0, 3, 0.5, 2, 4]);
  expect(
    getGPULocalKrigingParameterValues({
      covariance: {model: 'matern', smoothness: 0.5, range: 4, sill: 2}
    })[4]
  ).toBe(1);
  expect(
    getGPULocalKrigingParameterValues({
      covariance: {model: 'matern', smoothness: 2.5, range: 4, sill: 2}
    })[4]
  ).toBe(4);
});

it('packs co-kriging structures with B = L L^T after the structure rows', () => {
  const values = getGPULocalKrigingParameterValues({
    variableCount: 2,
    coregionalizationModels: [
      {covariance: {model: 'exponential', range: 1, sill: 2}, loadings: [1, 0, 0.5, 2]},
      {covariance: {model: 'gaussian', range: 3, sill: 1, nugget: 1}, loadings: [2, 0, 0, 1]}
    ]
  });
  expect(values.length).toBe(getGPULocalKrigingParameterLength(2, 2));
  expect(Array.from(values)).toEqual([
    8.5, 0, 0, 0, 1, 0, 2, 1, 2, 1, 1, 3, 1, 0.5, 0.5, 4.25, 4, 0, 0, 1
  ]);
});

it('getGPULocalKrigingParameterLength matches the univariate and co-kriging layouts', () => {
  expect(getGPULocalKrigingParameterLength()).toBe(GPU_LOCAL_KRIGING_PARAMETER_LENGTH);
  expect(getGPULocalKrigingParameterLength(4, 4)).toBe(4 + 16 + 64);
});

it('rejects invalid covariance settings and short targets', () => {
  expect(() => validatePredictionCovariance({model: 'matern', range: 1, sill: 1})).toThrow();
  expect(() =>
    validatePredictionCovariance({
      model: 'matern',
      smoothness: 2 as unknown as 1.5,
      range: 1,
      sill: 1
    })
  ).toThrow();
  expect(() =>
    getGPULocalKrigingParameterValues({covariance: {model: 'spherical', range: 0, sill: 1}})
  ).toThrow();
  expect(() =>
    getGPULocalKrigingParameterValues(
      {covariance: {model: 'spherical', range: 1, sill: 1}},
      new Float32Array(4)
    )
  ).toThrow();
  expect(() => getGPULocalKrigingParameterValues({})).toThrow();
});

it('the per-frame kernel reuses the one pivot-ratio limit and adds no second rule', () => {
  const source = getPerFramePredictionCovarianceWGSL(1, 1);
  expect(source).not.toContain('PIVOT');
  expect(getEquilibratedSolveWGSL(8)).toContain('SYSTEM_PIVOT_RATIO_TOLERANCE');
  expect(KRIGING_PIVOT_RATIO_TOLERANCE).toBe(1e-4);
});

it('requires covariance or a parameter buffer, and validates the parameter length', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const base = {
    positions: view('float32x2', 4),
    observations: view('float32x4', 4),
    supports: view('float32x2', 4),
    mode: 'ordinary' as const,
    output: {values: view('float32x4', 4), status: view('uint32', 4)}
  };
  expect(() => new GPULocalKriging(base)).toThrow('covariance or parameters');
  expect(
    () =>
      new GPULocalKriging({
        ...base,
        parameters: view('float32', GPU_LOCAL_KRIGING_PARAMETER_LENGTH - 1)
      })
  ).toThrow('parameters');
  expect(
    () =>
      new GPULocalKriging({
        ...base,
        parameters: view('float32', GPU_LOCAL_KRIGING_PARAMETER_LENGTH)
      })
  ).not.toThrow();
  expect(
    () =>
      new GPUConditionalGaussianSimulation({
        positions: base.positions,
        observations: base.observations,
        queries: view('float32x2', 4),
        mean: 0,
        seed: 1,
        replicateCount: 1,
        covariance: {model: 'matern', smoothness: 1.5, range: 1, sill: 1},
        output: {
          values: view('float32', 4),
          moments: view('float32x2', 4),
          status: view('uint32', 1)
        }
      })
  ).toThrow('Matern');
});
