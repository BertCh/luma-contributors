// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {fitVecchiaCovarianceFisherScoring} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-fisher-fit';

const initial = {model: 'exponential' as const, sill: 2, range: 4, nugget: 0.5};

it('N2 documents the conservative failure to resolve all required floor branches', async () => {
  let evaluationCount = 0;
  const result = await fitVecchiaCovarianceFisherScoring({
    initial: {model: 'matern-3/2', sill: 1, range: 4, nugget: 0.001},
    fitNugget: false,
    evaluate: () => {
      evaluationCount++;
      // Fixed nugget = floor. Decreasing sill: I00=0 cannot solve g0=2.
      // Increasing sill follows the floor: transformed I00=2 and g0=2 give a positive step.
      // The fitter deliberately requires both feasible branches before proceeding.
      return {
        logLikelihood: -7,
        score: [2, 0, 0],
        information: [0, 0, 0, 2, 0, 2],
        failedSiteCount: 0
      };
    }
  });
  expect(evaluationCount).toBe(1);
  expect(result.stallReason).toBe('direction-failure');
  expect(result.history).toEqual([]);
  expect(result.covariance).toEqual({model: 'matern-3/2', sill: 1, range: 4, nugget: 0.001});
  expect(Number.isFinite(result.logLikelihood)).toBe(true);
  expect(result.logLikelihood).toBe(-7);
  const guide = readFileSync('docs/api/gpu-spatial-analysis/vecchia-likelihood.md', 'utf8');
  expect(guide).toContain('resolve all required branches');
});

it('PTS-114 distinguishes unavailable information from a line search without discarding the initial fit', async () => {
  let evaluationCount = 0;
  const result = await fitVecchiaCovarianceFisherScoring({
    initial,
    evaluate: () => {
      evaluationCount++;
      // I = diag(2, 0, 1), g = [0, 1, 0]: row 1 requires 0 * step = 1.
      // No direction exists, so no trial can be evaluated or accepted.
      return {
        logLikelihood: -7,
        score: [0, 1, 0],
        information: [2, 0, 0, 0, 0, 1],
        failedSiteCount: 0
      };
    }
  });
  expect(evaluationCount).toBe(1);
  expect(result.stopReason).toBe('line-search-stall');
  expect(result.stallReason).toBe('direction-failure');
  expect(result.history).toEqual([]);
  expect(result.iterationCount).toBe(0);
  expect(result.covariance).toEqual(initial);
  expect(Number.isFinite(result.logLikelihood)).toBe(true);
  expect(result.logLikelihood).toBe(-7);
  expect(result.failed).toBe(false);
  expect(result.converged).toBe(false);
});

it('PTS-114 identifies rejected trial evaluations separately', async () => {
  let evaluationCount = 0;
  const result = await fitVecchiaCovarianceFisherScoring({
    initial,
    logLikelihoodNoise: 0,
    evaluate: () => ({
      // Identity information supports a direction, but every trial is strictly worse.
      logLikelihood: evaluationCount++ === 0 ? -7 : -100,
      score: [1, 0, 0],
      information: [1, 0, 0, 1, 0, 1],
      failedSiteCount: 0
    })
  });
  expect(evaluationCount).toBe(7);
  expect(result.stopReason).toBe('line-search-stall');
  expect(result.stallReason).toBe('line-search-rejected');
  expect(result.history).toEqual([]);
  expect(result.covariance).toEqual(initial);
  expect(Number.isFinite(result.logLikelihood)).toBe(true);
  expect(result.logLikelihood).toBe(-7);
});
