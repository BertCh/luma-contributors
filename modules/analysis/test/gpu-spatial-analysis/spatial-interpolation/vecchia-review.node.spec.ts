// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeVecchiaFisherScoringOnCPU} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation';
import {computeVecchiaLogLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation';
import {computeVecchiaSpaceTimeLogLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation';
import {getGPUVecchiaLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {getGPUVecchiaSpaceTimeLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';
import {
  OVER_WIDTH_EXPECTED,
  OVER_WIDTH_SCENE,
  TRANSLATED_VECCHIA_EXPECTED
} from './vecchia-review-fixtures';

const parameters = getGPUVecchiaLogLikelihoodParameterValues({
  model: 'exponential',
  sill: 1,
  range: 3
});

it('PTS-143 characterizes the CPU all-slot over-width rule independently of first-k and nearest-k', () => {
  const scene = OVER_WIDTH_SCENE;
  const input = {
    positions: Float32Array.from(scene.positions),
    observations: Float32Array.from(
      scene.values.flatMap((value, row) => [value, 0, 0, scene.validity[row]])
    ),
    neighborhoodOffsets: scene.offsets,
    neighborhoods: scene.neighbors,
    parameters
  };
  const likelihood = computeVecchiaLogLikelihoodOnCPU(input);
  const fisher = computeVecchiaFisherScoringOnCPU(input);
  const temporal = computeVecchiaSpaceTimeLogLikelihoodOnCPU({
    ...input,
    times: new Float32Array(5),
    parameters: getGPUVecchiaSpaceTimeLogLikelihoodParameterValues({
      type: 'metric',
      joint: {model: 'exponential', sill: 1, nugget: 0, range: 3},
      anisotropy: 1
    })
  });
  expect(temporal.conditionals).toHaveLength(20);
  expect(temporal.conditionalStatus).toHaveLength(5);
  expect(temporal.siteTerms).toHaveLength(5);
  expect(temporal.siteStatus).toHaveLength(5);
  expect(temporal.logLikelihood).toHaveLength(2);
  expect(temporal.status).toHaveLength(3);
  expect(Array.from(temporal.status)).toEqual([0, 1, 4]);
  expect(Array.from(likelihood.status)).toEqual([0, 1, 4]);
  expect(likelihood.conditionals).toHaveLength(20);
  expect(likelihood.conditionalStatus).toHaveLength(5);
  expect(likelihood.siteTerms).toHaveLength(5);
  expect(likelihood.siteStatus).toHaveLength(5);
  expect(likelihood.logLikelihood).toHaveLength(2);
  expect(likelihood.status).toHaveLength(3);
  expect(fisher.siteTerms).toHaveLength(50);
  expect(fisher.siteStatus).toHaveLength(5);
  expect(fisher.totals).toHaveLength(20);
  expect(fisher.status).toHaveLength(3);
  expect(Array.from(fisher.status)).toEqual([0, 1, 4]);
  for (const actual of [likelihood.siteTerms[4], temporal.siteTerms[4], fisher.siteTerms[40]]) {
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeCloseTo(OVER_WIDTH_EXPECTED.all.likelihood, 6);
    expect(Math.abs(actual - OVER_WIDTH_EXPECTED.first.likelihood)).toBeGreaterThan(0.1);
    expect(Math.abs(actual - OVER_WIDTH_EXPECTED.nearest.likelihood)).toBeGreaterThan(0.01);
  }
  expect(likelihood.conditionals[16]).toBeCloseTo(OVER_WIDTH_EXPECTED.all.mean, 6);
  expect(likelihood.conditionals[17]).toBeCloseTo(OVER_WIDTH_EXPECTED.all.variance, 6);
  expect(likelihood.conditionals[19]).toBe(3);
});

it('PTS-119 float validity excludes zero, negative and NaN in both CPU likelihood families and Fisher', () => {
  const input = {
    positions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0),
    times: new Float32Array(4),
    observations: Float32Array.of(2, 0, 0, 0.25, 100, 0, 0, 0, 200, 0, 0, -1, 300, 0, 0, NaN),
    neighborhoodOffsets: new Uint32Array(5),
    neighborhoods: new Uint32Array(0),
    parameters
  };
  const spaceTime = getGPUVecchiaSpaceTimeLogLikelihoodParameterValues({
    type: 'metric',
    joint: {model: 'exponential', sill: 1, nugget: 0, range: 3},
    anisotropy: 1
  });
  for (const result of [
    computeVecchiaLogLikelihoodOnCPU(input),
    computeVecchiaSpaceTimeLogLikelihoodOnCPU({...input, parameters: spaceTime}),
    computeVecchiaFisherScoringOnCPU(input)
  ]) {
    expect(Array.from(result.status)).toEqual([0, 3, 1]);
    expect(Array.from(result.siteStatus)).toEqual([0, 1, 1, 1]);
    // The only retained value is 2 under a unit-variance marginal N(0,1).
    expect(Number.isFinite(result.siteTerms[0])).toBe(true);
    expect(result.siteTerms[0]).toBeCloseTo(-2.9189385332046727, 6);
  }
});

it('PTS-139 translates spatial and space-time Vecchia conditionals without losing neighbors', () => {
  for (const [originX, originY] of [
    [0, 0],
    [500000, 5000000]
  ]) {
    const input = {
      positions: Float32Array.from([0, 1, 2].flatMap(x => [originX + x, originY])),
      times: new Float32Array(3),
      observations: Float32Array.of(1, 0, 0, 1, 2, 0, 0, 1, -1, 0, 0, 1),
      neighborhoodOffsets: Uint32Array.of(0, 0, 1, 3),
      neighborhoods: Uint32Array.of(0, 0, 1),
      parameters
    };
    const spaceTime = getGPUVecchiaSpaceTimeLogLikelihoodParameterValues({
      type: 'metric',
      joint: {model: 'exponential', sill: 1, nugget: 0, range: 3},
      anisotropy: 1
    });
    for (const result of [
      computeVecchiaLogLikelihoodOnCPU(input),
      computeVecchiaSpaceTimeLogLikelihoodOnCPU({...input, parameters: spaceTime})
    ]) {
      expect(Array.from(result.status)).toEqual([0, 0, 3]);
      expect(Number.isFinite(result.logLikelihood[0] + result.logLikelihood[1])).toBe(true);
      expect(result.logLikelihood[0] + result.logLikelihood[1]).toBeCloseTo(
        TRANSLATED_VECCHIA_EXPECTED.likelihood,
        8
      );
      for (let row = 0; row < 3; row++) {
        expect(Number.isFinite(result.conditionals[row * 4])).toBe(true);
        expect(Number.isFinite(result.conditionals[row * 4 + 1])).toBe(true);
        expect(result.conditionals[row * 4]).toBeCloseTo(TRANSLATED_VECCHIA_EXPECTED.means[row], 6);
        expect(result.conditionals[row * 4 + 1]).toBeCloseTo(
          TRANSLATED_VECCHIA_EXPECTED.variances[row],
          6
        );
        expect(result.conditionals[row * 4 + 3]).toBe(row);
      }
    }
  }
});
