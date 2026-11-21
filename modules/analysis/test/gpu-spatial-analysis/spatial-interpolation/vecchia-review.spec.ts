// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUVecchiaLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {getGPUVecchiaSpaceTimeLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';
import {
  createVecchiaLikelihoodHarness,
  createVecchiaFisherHarness,
  createVecchiaSpaceTimeHarness
} from './vecchia-likelihood-harness';
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
const spaceTimeParameters = getGPUVecchiaSpaceTimeLogLikelihoodParameterValues({
  type: 'metric',
  joint: {model: 'exponential', sill: 1, nugget: 0, range: 3},
  anisotropy: 1
});

it('PTS-143 characterizes first-usable-k Fisher/space-time and nearest-k likelihood on an unsorted over-width row', async () => {
  const likelihood = await createVecchiaLikelihoodHarness(OVER_WIDTH_SCENE);
  const fisher = await createVecchiaFisherHarness(OVER_WIDTH_SCENE);
  const spaceTime = await createVecchiaSpaceTimeHarness({
    ...OVER_WIDTH_SCENE,
    times: [0, 0, 0, 0, 0]
  });
  try {
    expect((await likelihood.evaluate(parameters)).failedSiteCount).toBe(0);
    expect((await fisher.evaluate(parameters)).failedSiteCount).toBe(0);
    expect((await spaceTime.evaluate(spaceTimeParameters)).failedSiteCount).toBe(0);
    const temporalRows = await spaceTime.readConditionals();
    expect(Array.from(temporalRows.siteStatus)).toEqual([1, 0, 0, 0, 0]);
    expect(Number.isFinite(temporalRows.conditionals[16])).toBe(true);
    expect(temporalRows.conditionals[16]).toBeCloseTo(OVER_WIDTH_EXPECTED.first.mean, 5);
    expect(temporalRows.conditionals[17]).toBeCloseTo(OVER_WIDTH_EXPECTED.first.variance, 5);
    expect(temporalRows.conditionals[19]).toBe(1);
    const rows = await likelihood.readConditionals();
    const scores = await fisher.readSites();
    expect(Array.from(rows.siteStatus)).toEqual([1, 0, 0, 0, 0]);
    expect(Array.from(scores.siteStatus)).toEqual([1, 0, 0, 0, 0]);
    expect(Number.isFinite(rows.siteTerms[4])).toBe(true);
    expect(Number.isFinite(scores.siteTerms[40])).toBe(true);
    expect(rows.siteTerms[4]).toBeCloseTo(OVER_WIDTH_EXPECTED.nearest.likelihood, 5);
    expect(scores.siteTerms[40]).toBeCloseTo(OVER_WIDTH_EXPECTED.first.likelihood, 5);
    expect(rows.conditionals[16]).toBeCloseTo(OVER_WIDTH_EXPECTED.nearest.mean, 5);
    expect(rows.conditionals[17]).toBeCloseTo(OVER_WIDTH_EXPECTED.nearest.variance, 5);
    expect(rows.conditionals[19]).toBe(1);
  } finally {
    spaceTime.destroy();
    likelihood.destroy();
    fisher.destroy();
  }
});

it('PTS-119 pins positive, zero, negative and NaN float observation validity in all Vecchia kernels', async () => {
  const scene = {
    positions: [0, 0, 1, 0, 2, 0, 3, 0],
    times: [0, 0, 0, 0],
    values: [2, 100, 200, 300],
    validity: [0.25, 0, -1, NaN],
    offsets: new Uint32Array(5),
    neighbors: new Uint32Array(0),
    neighborCount: 4
  };
  const likelihood = await createVecchiaLikelihoodHarness(scene);
  const fisher = await createVecchiaFisherHarness(scene);
  const spaceTime = await createVecchiaSpaceTimeHarness(scene);
  try {
    for (const result of [
      await likelihood.evaluate(parameters),
      await fisher.evaluate(parameters),
      await spaceTime.evaluate(spaceTimeParameters)
    ]) {
      expect(result.failedSiteCount).toBe(0);
      expect(result.usedSiteCount).toBe(1);
      expect(result.excludedSiteCount).toBe(3);
      expect(Number.isFinite(result.logLikelihood)).toBe(true);
      expect(result.logLikelihood).toBeCloseTo(-2.9189385332046727, 5);
    }
    for (const rows of [
      await likelihood.readConditionals(),
      await spaceTime.readConditionals(),
      await fisher.readSites()
    ]) {
      expect(Array.from(rows.siteStatus)).toEqual([0, 1, 1, 1]);
    }
  } finally {
    likelihood.destroy();
    fisher.destroy();
    spaceTime.destroy();
  }
});

it('PTS-139 preserves finite likelihood, variance and retained neighbors at projected origins', async () => {
  for (const [originX, originY] of [
    [0, 0],
    [500000, 5000000]
  ]) {
    const scene = {
      positions: [0, 1, 2].flatMap(x => [originX + x, originY]),
      times: [0, 0, 0],
      values: [1, 2, -1],
      offsets: Uint32Array.of(0, 0, 1, 3),
      neighbors: Uint32Array.of(0, 0, 1),
      neighborCount: 2
    };
    const likelihood = await createVecchiaLikelihoodHarness(scene);
    const fisher = await createVecchiaFisherHarness(scene);
    const spaceTime = await createVecchiaSpaceTimeHarness(scene);
    try {
      for (const result of [
        await likelihood.evaluate(parameters),
        await fisher.evaluate(parameters),
        await spaceTime.evaluate(spaceTimeParameters)
      ]) {
        expect(result.failedSiteCount).toBe(0);
        expect(Number.isFinite(result.logLikelihood)).toBe(true);
        expect(result.logLikelihood).toBeCloseTo(TRANSLATED_VECCHIA_EXPECTED.likelihood, 5);
        expect(result.usedSiteCount).toBe(3);
        expect(result.excludedSiteCount).toBe(0);
      }
      for (const rows of [
        await likelihood.readConditionals(),
        await spaceTime.readConditionals()
      ]) {
        expect(Array.from(rows.siteStatus)).toEqual([0, 0, 0]);
        expect(Array.from(rows.conditionalStatus)).toEqual([0, 0, 0]);
        for (let row = 0; row < 3; row++) {
          expect(Number.isFinite(rows.conditionals[row * 4])).toBe(true);
          expect(Number.isFinite(rows.conditionals[row * 4 + 1])).toBe(true);
          expect(rows.conditionals[row * 4]).toBeCloseTo(TRANSLATED_VECCHIA_EXPECTED.means[row], 5);
          expect(rows.conditionals[row * 4 + 1]).toBeCloseTo(
            TRANSLATED_VECCHIA_EXPECTED.variances[row],
            5
          );
          expect(rows.conditionals[row * 4 + 3]).toBe(row);
        }
      }
    } finally {
      likelihood.destroy();
      fisher.destroy();
      spaceTime.destroy();
    }
  }
});
