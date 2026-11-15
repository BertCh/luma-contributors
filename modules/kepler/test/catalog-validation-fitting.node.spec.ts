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
    siteTerms: new Float32Array(4),
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
