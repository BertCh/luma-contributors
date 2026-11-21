// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {computeVecchiaFisherScoringOnCPU} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation';
import {getGPUVecchiaLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {getVecchiaFisherEvaluation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring';
import {describe, expect, it} from 'vitest';
import {
  fitVecchiaCovarianceFisherScoring,
  type VecchiaFisherEvaluation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-fisher-fit';

const initial = {model: 'exponential' as const, sill: 1, range: 2, nugget: 0};
const stationary: VecchiaFisherEvaluation = {
  logLikelihood: -2,
  score: [0, 0, 0],
  information: [1, 0, 0, 1, 0, 1],
  failedSiteCount: 0
};

describe('Fisher constrained steps and exits (PTS-115, PTS-114, PTS-143)', () => {
  it.each([0, 0.1])('carries only a sill-dependent effective nugget: fixed %s', async nugget => {
    // Independent log-coordinate objective L = log(sill) + 2 log(nugget).
    // On nugget = .001*sill, dL/dlog(sill) = 3; above it with nugget fixed, 1.
    // I = diag(2,1,4), so the constrained sill information is 6 (or 2 off floor).
    const likelihood = (sill: number) =>
      Math.log(sill) + 2 * Math.log(Math.max(nugget, 0.001 * sill));
    const difference = (likelihood(Math.exp(1e-5)) - likelihood(Math.exp(-1e-5))) / 2e-5;
    expect(difference).toBeCloseTo(nugget === 0 ? 3 : 1, 8);
    const result = await fitVecchiaCovarianceFisherScoring({
      initial: {...initial, model: 'matern-3/2', nugget},
      fitNugget: false,
      maximumIterations: 1,
      evaluate: covariance => ({
        ...stationary,
        logLikelihood: likelihood(covariance.sill),
        score: [1, 0, 2],
        information: [2, 0, 0, 1, 0, 4]
      })
    });
    expect(Number.isFinite(result.history[0].decrement)).toBe(true);
    expect(result.history[0].decrement).toBeCloseTo(nugget === 0 ? 1.5 : 0.5, 8);
    expect(Math.log(result.covariance.sill)).toBeCloseTo(0.5, 8);
    expect(result.covariance.nugget).toBeCloseTo(nugget === 0 ? 0.001 * Math.exp(0.5) : 0.1, 10);
  });

  it('reports immediate convergence without an accepted history row', async () => {
    let calls = 0;
    const result = await fitVecchiaCovarianceFisherScoring({
      initial,
      evaluate: () => {
        calls++;
        return stationary;
      }
    });
    expect(calls).toBe(1);
    expect(result.covariance).toEqual(initial);
    expect(result.logLikelihood).toBe(-2);
    expect(result.failed).toBe(false);
    expect(result.iterationLimitReached).toBe(false);
    expect(result.stopReason).toBe('converged');
    expect(result.converged).toBe(true);
    expect(result.iterationCount).toBe(0);
    expect(result.history).toEqual([]);
  });

  it('reports initial failure without proposals', async () => {
    let calls = 0;
    const result = await fitVecchiaCovarianceFisherScoring({
      initial,
      evaluate: () => {
        calls++;
        return {...stationary, failedSiteCount: 1};
      }
    });
    expect(calls).toBe(1);
    expect(result.stopReason).toBe('initial-failure');
    expect(result.failed).toBe(true);
    expect(result.iterationLimitReached).toBe(false);
    expect(result.converged).toBe(false);
    expect(result.iterationCount).toBe(0);
    expect(result.history).toEqual([]);
    expect(Number.isNaN(result.logLikelihood)).toBe(true);
    expect(result.covariance).toEqual(initial);
  });

  it.each([0, 2])('reports an exhausted budget of %s accepted steps', async maximumIterations => {
    let calls = 0;
    const result = await fitVecchiaCovarianceFisherScoring({
      initial,
      maximumIterations,
      evaluate: covariance => {
        calls++;
        return {...stationary, logLikelihood: Math.log(covariance.sill), score: [1, 0, 0]};
      }
    });
    expect(result.stopReason).toBe('iteration-limit');
    expect(result.iterationLimitReached).toBe(true);
    expect(result.failed).toBe(false);
    expect(result.converged).toBe(false);
    expect(result.iterationCount).toBe(maximumIterations);
    expect(result.history).toHaveLength(maximumIterations);
    expect(calls).toBe(maximumIterations + 1);
    result.history.forEach((entry, row) => {
      expect(Number.isFinite(entry.logLikelihood)).toBe(true);
      expect(entry.logLikelihood).toBeCloseTo(row + 1, 8);
    });
    expect(Math.log(result.covariance.sill)).toBeCloseTo(maximumIterations, 8);
    expect(result.logLikelihood).toBeCloseTo(maximumIterations, 8);
  });

  it.each([
    0, 1
  ])('keeps the last fit when every proposal stalls after %s steps', async acceptedSteps => {
    let calls = 0;
    const result = await fitVecchiaCovarianceFisherScoring({
      initial,
      logLikelihoodNoise: 0,
      evaluate: () => {
        calls++;
        return {
          ...stationary,
          logLikelihood: calls <= acceptedSteps + 1 ? calls : -100,
          score: [1, 0, 0]
        };
      }
    });
    expect(result.stopReason).toBe('line-search-stall');
    expect(result.converged).toBe(false);
    expect(result.failed).toBe(false);
    expect(result.iterationLimitReached).toBe(false);
    expect(result.iterationCount).toBe(acceptedSteps);
    expect(result.history).toHaveLength(acceptedSteps);
    expect(result.history.map(entry => entry.logLikelihood)).toEqual(
      acceptedSteps === 0 ? [] : [2]
    );
    expect(result.logLikelihood).toBe(acceptedSteps + 1);
    expect(Math.log(result.covariance.sill)).toBeCloseTo(acceptedSteps, 8);
    expect(calls).toBe(acceptedSteps + 7);
  });
});

it('B2 checks the downward fixed-nugget branch at equality with the Matern floor', async () => {
  const proposals: {sill: number; nugget?: number}[] = [];
  const result = await fitVecchiaCovarianceFisherScoring({
    initial: {model: 'matern-3/2', sill: 4, range: 2, nugget: 0.004},
    fitNugget: false,
    maximumIterations: 1,
    logLikelihoodNoise: 0,
    evaluate: covariance => {
      proposals.push(covariance);
      return {
        ...stationary,
        logLikelihood: -Math.log(covariance.sill / 4) + Math.log(covariance.nugget! / 0.004),
        score: [-1, 0, 1]
      };
    }
  });
  // Independent feasible left derivative: nugget stays .004, so dL/dlog(sill)=-1.
  expect(proposals).toHaveLength(2);
  expect(result.converged).toBe(false);
  expect(result.iterationCount).toBe(1);
  expect(Number.isFinite(result.covariance.sill)).toBe(true);
  expect(result.covariance.sill).toBeCloseTo(1.4715177646857693, 7); // 4/e.
  expect(result.covariance.nugget).toBe(0.004);
  expect(result.logLikelihood).toBeCloseTo(1, 8);
  expect(result.history[0].logLikelihood).toBe(result.logLikelihood);
});

it('B3 moves a pinned estimated nugget to the trial floor for a downward sill step', async () => {
  const proposals: {sill: number; nugget?: number}[] = [];
  const result = await fitVecchiaCovarianceFisherScoring({
    initial: {model: 'matern-3/2', sill: 4, range: 2, nugget: 0.004},
    maximumIterations: 1,
    logLikelihoodNoise: 0,
    evaluate: covariance => {
      proposals.push(covariance);
      return {
        ...stationary,
        logLikelihood: -Math.log(covariance.nugget! / 0.004),
        score: [0, 0, -1]
      };
    }
  });
  // Constrained score=-1, information=2: log sill and log nugget both move by -1/2.
  expect(proposals).toHaveLength(2);
  expect(result.iterationCount).toBe(1);
  expect(result.history).toHaveLength(1);
  expect(result.stopReason).toBe('iteration-limit');
  expect(proposals[1].sill).toBeCloseTo(2.4261226388505337, 7);
  expect(proposals[1].nugget).toBeCloseTo(0.0024261226388505337, 10);
  expect(proposals[1].nugget).toBe(0.001 * proposals[1].sill);
  expect(result.logLikelihood).toBeCloseTo(0.5, 8);
  expect(result.history[0].logLikelihood).toBeCloseTo(0.5, 8);
});

it('B4 fits marginal-only observations through the public packer with unidentifiable range', async () => {
  let calls = 0;
  const result = await fitVecchiaCovarianceFisherScoring({
    initial: {model: 'exponential', sill: 2, range: 3, nugget: 0},
    convergenceTolerance: 1e-12,
    evaluate: covariance => {
      calls++;
      const rows = computeVecchiaFisherScoringOnCPU({
        positions: Float32Array.of(0, 0, 1, 0),
        observations: Float32Array.of(2, 0, 0, 1, 3, 0, 0, 1),
        neighborhoodOffsets: Uint32Array.of(0, 0, 0),
        neighborhoods: new Uint32Array(0),
        parameters: getGPUVecchiaLogLikelihoodParameterValues(covariance)
      });
      return getVecchiaFisherEvaluation(rows.totals, rows.status);
    }
  });
  // Independent N(0,sill) MLE: (2^2+3^2)/2 = 6.5. The likelihood is independent of range.
  expect(calls).toBeGreaterThan(1);
  expect(result.stopReason).toBe('converged');
  expect(Number.isFinite(result.covariance.sill)).toBe(true);
  expect(result.covariance.sill).toBeCloseTo(6.5, 4);
  expect(result.covariance.range).toBeCloseTo(3, 12);
  expect(Number.isFinite(result.logLikelihood)).toBe(true);
  expect(result.logLikelihood).toBeCloseTo(-4.709679243310936, 9);
  expect(result.history).toHaveLength(result.iterationCount);
});

it('B4 preserves the finite fit without proposals when the information cannot support its score', async () => {
  let calls = 0;
  const result = await fitVecchiaCovarianceFisherScoring({
    initial,
    evaluate: () => {
      calls++;
      return {...stationary, score: [1, 0, 0], information: [0, 0, 0, 0, 0, 0]};
    }
  });
  expect(calls).toBe(1);
  expect(result.stopReason).toBe('line-search-stall');
  expect(result.covariance).toEqual(initial);
  expect(result.logLikelihood).toBe(-2);
  expect(result.history).toEqual([]);
});

it('S2 carries nonzero sill/nugget/range information cross-terms along the floor', async () => {
  const result = await fitVecchiaCovarianceFisherScoring({
    initial: {model: 'matern-3/2', sill: 1, range: 2, nugget: 0},
    fitNugget: false,
    maximumIterations: 1,
    evaluate: covariance => ({
      ...stationary,
      score: [1, 2, 3],
      information: [4, 1, 2, 3, 0.5, 5],
      logLikelihood:
        Math.log(covariance.sill) +
        2 * Math.log(covariance.range / 2) +
        3 * Math.log(covariance.nugget! / 0.001)
    })
  });
  // J'I J=[[13,1.5],[1.5,3]], J'g=[4,2]. Its inverse gives [12/49,80/147], decrement 304/147.
  expect(Number.isFinite(result.history[0].decrement)).toBe(true);
  expect(result.history[0].decrement).toBeCloseTo(2.068027210884354, 8);
  expect(Math.log(result.covariance.sill)).toBeCloseTo(12 / 49, 8);
  expect(Math.log(result.covariance.range / 2)).toBeCloseTo(80 / 147, 8);
  expect(result.history[0].logLikelihood).toBeCloseTo(2.068027210884354, 8);
});
