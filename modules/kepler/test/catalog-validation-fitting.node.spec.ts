// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {afterAll, beforeAll, beforeEach, expect, test, vi} from 'vitest';
import {getGPUVecchiaSpaceTimeLogLikelihoodParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
// Keep the actual library optimizers and parameter packers; replace only GPU execution/readback.
// Node specs share a module cache, so scope the replacement to this file's dynamic imports.
const gpuSession = {evaluate: vi.fn()};
let fitting: typeof import('../src/analysis/catalog-validation-vecchia');
let variogram: typeof import('../src/analysis/kriging-variogram-fit');
beforeAll(async () => {
  vi.resetModules();
  vi.doMock('../src/analysis/catalog-gpu-session', async () => ({
    ...(await vi.importActual<typeof import('../src/analysis/catalog-gpu-session')>(
      '../src/analysis/catalog-gpu-session'
    )),
    withCatalogGpuSession: gpuSession.evaluate
  }));
  fitting = await import('../src/analysis/catalog-validation-vecchia');
  variogram = await import('../src/analysis/kriging-variogram-fit');
});
afterAll(() => {
  vi.doUnmock('../src/analysis/catalog-gpu-session');
  vi.resetModules();
});

beforeEach(() => {
  gpuSession.evaluate.mockReset();
  gpuSession.evaluate.mockResolvedValue({
    logLikelihood: -1,
    score: [0, 0, 0],
    information: [1, 0, 0, 1, 0, 1],
    failedSiteCount: 0,
    excludedSiteCount: 0,
    usedSiteCount: 4,
    siteTerms: new Float32Array(40),
    siteStatus: new Uint32Array(4)
  });
});

const temporalData = {
  positions: Float32Array.of(0, 0, 1, 0, 0, 1, 1, 1),
  observations: Float32Array.of(1, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 1, 4, 0, 0, 1),
  neighborhoods: {offsets: Uint32Array.of(0, 0, 1, 2, 3), neighbors: Uint32Array.of(0, 0, 1)},
  neighborCount: 1,
  times: Float32Array.of(0, 1, 2, 3),
  temporalContext: {
    unit: 'second' as const,
    epoch: 0n,
    encoding: 'float32-relative' as const,
    axis: {kind: 'irregular' as const}
  },
  anisotropy: 1
};

test('actual space-time Matern optimizer admits a zero initial nugget using the total-sill floor', async () => {
  const fit = await fitting.fitValidationVecchia(
    {} as Device,
    temporalData,
    {model: 'matern-3/2', sill: 1, range: 100, nugget: 0},
    'likelihood',
    8
  );
  expect(fit.failed).toBe(false);
  expect(gpuSession.evaluate).toHaveBeenCalledTimes(4);
  expect(fit.covariance.nugget!).toBeGreaterThan(0.001 * fit.covariance.sill);
  expect(fit.covariance.nugget! / (fit.covariance.sill + fit.covariance.nugget!)).toBeCloseTo(
    0.001,
    15
  );
  expect(() =>
    getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(
      fitting.makeValidationSpaceTimeCovariance(fit.covariance, 1)
    )
  ).not.toThrow();
});

test('actual space-time optimizer penalizes inadmissible packed proposals without aborting or dispatching', async () => {
  const fit = await fitting.fitValidationVecchia(
    {} as Device,
    {...temporalData, anisotropy: NaN},
    {model: 'matern-3/2', sill: 1, range: 100, nugget: 0},
    'likelihood',
    4
  );
  expect(fit.failed).toBe(true);
  expect(gpuSession.evaluate).not.toHaveBeenCalled();
});

test.each([
  1, -1
])('Vecchia retains the empirical fitter population near %s times 86 degrees and at the pole', async sign => {
  const longitudes = [0, 0.1, 0, 0.1, 0.05, 0, 181];
  const latitudes = [86, 86, 86.1, 86.1, 90, 91, 0].map(latitude => latitude * sign);
  const response = [1, 2, 3, 4, 5, 99, 99];
  const empirical = variogram.estimateKeplerEmpiricalVariogram(longitudes, latitudes, response);
  const fit = await variogram.fitKeplerVecchiaVariogram(
    {} as Device,
    longitudes,
    latitudes,
    response,
    {
      method: 'likelihood',
      initial: {model: 'exponential', sill: 1, range: 100, nugget: 0.05},
      maximumIterations: 4
    }
  );
  expect(empirical.validRowCount).toBe(5);
  expect(fit.diagnostics!['valid_rows']).toBe(empirical.validRowCount);
  expect(fit.diagnostics!['excluded_rows']).toBe(2);
  expect(gpuSession.evaluate).toHaveBeenCalled();
});

test.each([
  'line-search-stall',
  'iteration-limit'
] as const)('Fisher adapter retains %s and accepted evidence', async reason => {
  let calls = 0;
  gpuSession.evaluate.mockImplementation(async () => ({
    logLikelihood: reason === 'line-search-stall' && calls > 1 ? -100 : calls++,
    score: [1, 0, 0],
    information: [1, 0, 0, 1, 0, 1],
    failedSiteCount: 0,
    excludedSiteCount: 0,
    usedSiteCount: 4,
    siteTerms: new Float32Array(40),
    siteStatus: new Uint32Array(4)
  }));
  const {times: _times, temporalContext: _context, ...data} = temporalData;
  const fit = await fitting.fitValidationVecchia(
    {} as Device,
    data,
    {model: 'exponential', sill: 1, range: 2, nugget: 0.1},
    'fisher',
    4
  );
  expect(fit.stopReason).toBe(reason);
  expect(fit.stallReason).toBe(reason === 'line-search-stall' ? 'line-search-rejected' : undefined);
  expect(fit.failed).toBe(false);
  expect(fit.converged).toBe(false);
  expect(fit.iterationLimitReached).toBe(reason === 'iteration-limit');
  const accepted = reason === 'line-search-stall' ? 1 : 4;
  expect(fit.budgetUsed).toBe(accepted);
  expect(fit.history?.map(entry => entry.logLikelihood)).toEqual(
    accepted === 1 ? [1] : [1, 2, 3, 4]
  );
  expect(fit.logLikelihood).toBe(accepted);
  // Diagonal information=I and score=(1,0,0): each accepted log-sill step is 1
  // up to the producer's 1e-9 diagonal stabilizer; rejected steps must never be returned.
  expect(Number.isFinite(fit.covariance.sill)).toBe(true);
  expect(Math.log(fit.covariance.sill)).toBeCloseTo(accepted, 6);
  expect(fit.covariance.range).toBe(2);
  expect(fit.covariance.nugget).toBeCloseTo(0.1, 12);
  // Reset the deterministic sequence and exercise the ordinary-kriging consumer too.
  calls = 0;
  const kriging = await variogram.fitKeplerVecchiaVariogram(
    {} as Device,
    [0, 0.1, 0, 0.1],
    [0, 0, 0.1, 0.1],
    [1, 2, 3, 4],
    {
      method: 'fisher',
      initial: {model: 'exponential', sill: 1, range: 2, nugget: 0.1},
      maximumIterations: 4
    }
  );
  expect(kriging.stopReason).toBe(reason);
  expect(kriging.stallReason).toBe(
    reason === 'line-search-stall' ? 'line-search-rejected' : undefined
  );
  expect(kriging.nonconvergenceDiagnostic).toContain(reason);
  expect(kriging.iterationLimitReached).toBe(reason === 'iteration-limit');
  expect(kriging.history?.map(entry => entry.logLikelihood)).toEqual(
    accepted === 1 ? [1] : [1, 2, 3, 4]
  );
  expect(kriging.logLikelihood).toBe(accepted);
  expect(Number.isFinite(kriging.sill)).toBe(true);
  expect(Math.log(kriging.sill)).toBeCloseTo(accepted, 6);
  calls = 0;
  const binding = fitting.VALIDATION_VECCHIA_BINDINGS.find(
    entry => entry.id === 'validation-vecchia-fisher'
  )!;
  const published = await binding.runGpu!(
    {device: {} as Device},
    {
      points: {
        type: 'FeatureCollection',
        features: [0, 1, 2, 3].map(index => ({
          type: 'Feature' as const,
          geometry: {type: 'Point' as const, coordinates: [index, 0]},
          properties: {value: index + 1}
        }))
      }
    },
    {
      fit: true,
      maximumIterations: 4,
      model: 'exponential',
      sill: 1,
      range: 2,
      nugget: 0.1,
      neighborCount: 1,
      mean: 0,
      coordinateSystem: 'planar',
      valueProperty: 'value',
      errorVarianceProperty: ''
    }
  );
  expect(published.summary['stop_reason']).toBe(reason);
  expect(
    published.diagnostics?.filter(entry => entry.code === 'vecchia-nonconvergence')
  ).toHaveLength(1);
  expect(published.diagnostics).toContainEqual(
    expect.objectContaining({
      code: 'vecchia-nonconvergence',
      severity: 'warning',
      count: 1,
      message: expect.stringContaining(reason)
    })
  );
  expect(published.diagnostics?.at(-1)?.message).toContain(`${accepted} accepted steps`);
  expect(published.tables[0].columns['status']).toEqual([0, 0, 0, 0]);
});

test('initial Fisher failure reports no accepted fit and preserves its reason in kriging errors', async () => {
  gpuSession.evaluate.mockResolvedValue({
    logLikelihood: NaN,
    score: [NaN, NaN, NaN],
    information: [NaN, NaN, NaN, NaN, NaN, NaN],
    failedSiteCount: 4,
    excludedSiteCount: 0,
    usedSiteCount: 0,
    siteTerms: new Float32Array(40),
    siteStatus: new Uint32Array(4).fill(2)
  });
  const binding = fitting.VALIDATION_VECCHIA_BINDINGS.find(
    entry => entry.id === 'validation-vecchia-fisher'
  )!;
  const published = await binding.runGpu!(
    {device: {} as Device},
    {
      points: {
        type: 'FeatureCollection',
        features: [0, 1, 2, 3].map(index => ({
          type: 'Feature' as const,
          geometry: {type: 'Point' as const, coordinates: [index, 0]},
          properties: {value: index + 1}
        }))
      }
    },
    {
      fit: true,
      maximumIterations: 4,
      model: 'exponential',
      sill: 1,
      range: 2,
      nugget: 0.1,
      neighborCount: 1,
      mean: 0,
      coordinateSystem: 'planar',
      valueProperty: 'value',
      errorVarianceProperty: ''
    }
  );
  expect(published.summary['stop_reason']).toBe('initial-failure');
  expect(
    published.diagnostics?.filter(entry => entry.code === 'vecchia-initial-failure')
  ).toHaveLength(1);
  expect(published.diagnostics).toContainEqual(
    expect.objectContaining({
      code: 'vecchia-initial-failure',
      severity: 'error',
      count: 1,
      message: expect.stringContaining('no usable fit')
    })
  );
  expect(
    published.diagnostics?.some(diagnostic => diagnostic.message.includes('retained last accepted'))
  ).toBe(false);
  await expect(
    variogram.fitKeplerVecchiaVariogram(
      {} as Device,
      [0, 0.1, 0, 0.1],
      [0, 0, 0.1, 0.1],
      [1, 2, 3, 4],
      {
        method: 'fisher',
        initial: {model: 'exponential', sill: 1, range: 2, nugget: 0.1},
        maximumIterations: 4
      }
    )
  ).rejects.toThrow('initial-failure');
});

test('converged Fisher fit appends no nonconvergence diagnostic', async () => {
  const binding = fitting.VALIDATION_VECCHIA_BINDINGS.find(
    entry => entry.id === 'validation-vecchia-fisher'
  )!;
  const published = await binding.runGpu!(
    {device: {} as Device},
    {
      points: {
        type: 'FeatureCollection',
        features: [0, 1, 2, 3].map(index => ({
          geometry: {type: 'Point' as const, coordinates: [index, 0]},
          properties: {value: index + 1}
        }))
      }
    },
    {
      fit: true,
      maximumIterations: 4,
      model: 'exponential',
      sill: 1,
      range: 2,
      nugget: 0.1,
      neighborCount: 1,
      mean: 0,
      coordinateSystem: 'planar',
      valueProperty: 'value',
      errorVarianceProperty: ''
    }
  );
  // Zero score and identity information give a stationary initial covariance.
  const metrics = published.tables[1];
  expect(metrics.columns['value'][metrics.columns['metric'].indexOf('converged')]).toBe(1);
  expect(published.diagnostics).toEqual([]);
  for (const [name, column] of Object.entries(published.tables[0].columns)) {
    if (name === 'feature_index' || name === 'geometry') continue;
    expect(column, name).toHaveLength(4);
    expect(column, name).toEqual([0, 0, 0, 0]);
  }
});
