// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, describe, expect, it} from 'vitest';
import type {SpaceTimeVariogramModel} from '../../../src/dataframe-analysis/pair-statistics/space-time-variogram-model';
import {getGPUVecchiaSpaceTimeLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';
import {buildVecchiaConditioningSets} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-conditioning-sets';
import {maximizeVecchiaLogLikelihood} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-fisher-fit';
import {getReferenceSpaceTimeCovariance} from './space-time-kriging-reference';
import {
  createExperimentGaussian,
  createExperimentRandom,
  evaluateReferenceConditionalLogLikelihood,
  simulateReferenceConditionalField
} from './vecchia-likelihood-reference';
import {createVecchiaSpaceTimeHarness} from './vecchia-likelihood-harness';

const TIME_SCALE = 500; // metres per second of the neighbour metric

function createScene(rowCount: number, neighborCount: number, seed: number) {
  const random = createExperimentRandom(seed);
  // Rows sorted by time: time ascending is the Vecchia order, row order is the same.
  const order = Array.from({length: rowCount}, () => random() * 200).sort(
    (first, second) => first - second
  );
  const times = Float64Array.from(order, value => Math.fround(value));
  const positions = Float64Array.from({length: rowCount * 2}, () =>
    Math.fround((random() - 0.5) * 100000)
  );
  const {offsets, neighbors} = buildVecchiaConditioningSets({
    positions,
    neighborCount,
    times,
    timeScale: TIME_SCALE
  });
  return {
    positions,
    times,
    offsets,
    neighbors,
    neighborCount,
    gaussian: createExperimentGaussian(random)
  };
}

function getCovarianceOf(
  model: SpaceTimeVariogramModel,
  scene: {positions: ArrayLike<number>; times: ArrayLike<number>}
) {
  return (first: number, second: number): number =>
    getReferenceSpaceTimeCovariance(
      model,
      Math.hypot(
        scene.positions[first * 2] - scene.positions[second * 2],
        scene.positions[first * 2 + 1] - scene.positions[second * 2 + 1]
      ),
      Math.abs(scene.times[first] - scene.times[second])
    );
}

const FAMILIES: SpaceTimeVariogramModel[] = [
  {
    type: 'separable',
    sill: 2,
    spatial: {model: 'exponential', nugget: 0.05, sill: 0.95, range: 40000},
    temporal: {model: 'exponential', nugget: 0.05, sill: 0.95, range: 80}
  },
  {
    type: 'product-sum',
    k: 0.4,
    spatial: {model: 'matern-3/2', nugget: 0.05, sill: 1, range: 25000},
    temporal: {model: 'exponential', nugget: 0.02, sill: 0.8, range: 60}
  },
  {
    type: 'metric',
    joint: {model: 'exponential', nugget: 0.05, sill: 1.5, range: 50000},
    anisotropy: TIME_SCALE
  },
  {
    type: 'sum-metric',
    spatial: {model: 'exponential', nugget: 0.02, sill: 0.6, range: 30000},
    temporal: {model: 'exponential', nugget: 0.01, sill: 0.3, range: 70},
    joint: {model: 'exponential', nugget: 0.03, sill: 0.8, range: 40000},
    anisotropy: TIME_SCALE
  },
  {
    type: 'gneiting',
    nugget: 0.05,
    sill: 1.5,
    spatialRange: 28000,
    temporalRange: 90,
    alpha: 0.5,
    gamma: 0.5,
    beta: 0.6,
    tau: 1
  }
];

describe('GPUVecchiaSpaceTimeLogLikelihood', () => {
  const destroyers: (() => void)[] = [];
  afterEach(() => destroyers.splice(0).forEach(destroy => destroy()));

  it('matches the binary64 reference across space-time families without a recompile', async () => {
    const scene = createScene(900, 16, 77);
    const values = simulateReferenceConditionalField(
      scene.gaussian,
      scene.offsets,
      scene.neighbors,
      getCovarianceOf(FAMILIES[0], scene)
    );
    const harness = await createVecchiaSpaceTimeHarness({...scene, values});
    destroyers.push(harness.destroy);
    for (const model of FAMILIES) {
      const reference = evaluateReferenceConditionalLogLikelihood(
        values,
        scene.offsets,
        scene.neighbors,
        getCovarianceOf(model, scene)
      );
      const result = await harness.evaluate(
        getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, undefined, TIME_SCALE)
      );
      expect(result.failedSiteCount).toBe(0);
      expect(Math.abs(result.logLikelihood - reference)).toBeLessThan(0.03);
    }
    expect(harness.buildCount).toBe(1);
  });

  it('fits sill and ranges through per-frame parameters on the CPU', async () => {
    const scene = createScene(1500, 16, 3);
    const truth = FAMILIES[0] as Extract<SpaceTimeVariogramModel, {type: 'separable'}>;
    const values = simulateReferenceConditionalField(
      scene.gaussian,
      scene.offsets,
      scene.neighbors,
      getCovarianceOf(truth, scene)
    );
    const harness = await createVecchiaSpaceTimeHarness({...scene, values});
    destroyers.push(harness.destroy);
    const toModel = ([
      logSill,
      logSpatial,
      logTemporal
    ]: readonly number[]): SpaceTimeVariogramModel => ({
      type: 'separable',
      sill: Math.exp(logSill),
      spatial: {...truth.spatial, range: Math.exp(logSpatial)},
      temporal: {...truth.temporal, range: Math.exp(logTemporal)}
    });
    const initial = [
      Math.log(truth.sill * 0.6),
      Math.log(truth.spatial.range * 1.7),
      Math.log(truth.temporal.range * 0.5)
    ];
    const gpuFit = await maximizeVecchiaLogLikelihood({
      initial,
      evaluate: async point => {
        const model = toModel(point);
        return (
          await harness.evaluate(
            getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, undefined, TIME_SCALE)
          )
        ).logLikelihood;
      }
    });
    const referenceFit = await maximizeVecchiaLogLikelihood({
      initial,
      evaluate: point =>
        evaluateReferenceConditionalLogLikelihood(
          values,
          scene.offsets,
          scene.neighbors,
          getCovarianceOf(toModel(point), scene)
        )
    });
    expect(gpuFit.converged).toBe(true);
    expect(harness.buildCount).toBe(1);
    expect(Math.abs(gpuFit.logLikelihood - referenceFit.logLikelihood)).toBeLessThan(0.1);
    gpuFit.values.forEach((value, index) => {
      expect(Math.abs(value - referenceFit.values[index])).toBeLessThan(0.05);
    });
  });

  it('rejects a Matern 3/2 structure with no nugget', () => {
    expect(() =>
      getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(
        {
          type: 'metric',
          joint: {model: 'matern-3/2', nugget: 0, sill: 1.5, range: 3},
          anisotropy: 1
        },
        undefined,
        1
      )
    ).toThrow(/at least 0.001/);
  });
});
