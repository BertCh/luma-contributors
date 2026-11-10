// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  evaluateSpaceTimeVariogramModel,
  fitSpaceTimeVariogramModel,
  getGPUSpaceTimeVariogramBinLags,
  type SpaceTimeVariogramModel
} from '../../../src/dataframe-analysis/pair-statistics/space-time-variogram-model';
import {
  evaluateVariogramModel,
  fitVariogramModel,
  getVariogramModelShape
} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';

const SURFACE = {lagCount: 10, temporalLagCount: 8, maximumDistance: 10, maximumTimeLag: 24};
const {distances, timeLags} = getGPUSpaceTimeVariogramBinLags(SURFACE);

function createSurface(model: SpaceTimeVariogramModel) {
  const semivariances = Float64Array.from(distances, (distance, bin) =>
    evaluateSpaceTimeVariogramModel(model, distance, timeLags[bin])
  );
  const pairCounts = Float64Array.from(distances, () => 500);
  return {distances, timeLags, semivariances, pairCounts};
}

function expectFitReproducesSurface(model: SpaceTimeVariogramModel, weighting: 'pairs' | 'none') {
  const surface = createSurface(model);
  const fit = fitSpaceTimeVariogramModel(surface, {type: model.type, weighting});
  distances.forEach((distance, bin) => {
    if (distance === 0 && timeLags[bin] === 0) {
      return;
    }
    const expected = surface.semivariances[bin];
    expect(
      Math.abs(evaluateSpaceTimeVariogramModel(fit.model, distance, timeLags[bin]) - expected),
      `bin ${bin}`
    ).toBeLessThan(2e-3 * Math.max(expected, 0.5));
  });
  return fit;
}

describe('fitSpaceTimeVariogramModel', () => {
  it('bin lags follow the zero-lag classes and midpoints', () => {
    const lags = getGPUSpaceTimeVariogramBinLags({
      lagCount: 2,
      temporalLagCount: 3,
      maximumDistance: 4,
      maximumTimeLag: 6
    });
    expect(Array.from(lags.distances.slice(0, 4))).toEqual([0, 0, 0, 0]);
    expect(Array.from(lags.timeLags.slice(0, 4))).toEqual([0, 1, 3, 5]);
    expect(lags.distances[4]).toBe(1);
    expect(lags.distances[8]).toBe(3);
  });

  it('recovers a separable surface', () => {
    const fit = expectFitReproducesSurface(
      {
        type: 'separable',
        sill: 3,
        spatial: {model: 'exponential', nugget: 0.1, sill: 0.9, range: 4},
        temporal: {model: 'exponential', nugget: 0.05, sill: 0.95, range: 9}
      },
      'pairs'
    );
    if (fit.model.type !== 'separable') throw new Error('type');
    expect(fit.model.sill).toBeCloseTo(3, 2);
    expect(fit.model.spatial.range).toBeCloseTo(4, 1);
    expect(fit.model.temporal.range).toBeCloseTo(9, 1);
  });

  it('recovers a product-sum surface, including k', () => {
    const fit = expectFitReproducesSurface(
      {
        type: 'product-sum',
        k: 0.4,
        spatial: {model: 'exponential', nugget: 0.1, sill: 1.4, range: 5},
        temporal: {model: 'exponential', nugget: 0.05, sill: 0.8, range: 12}
      },
      'pairs'
    );
    if (fit.model.type !== 'product-sum') throw new Error('type');
    expect(fit.model.k).toBeCloseTo(0.4, 1);
    expect(fit.model.spatial.range).toBeCloseTo(5, 0);
  });

  it('recovers a metric surface, including the anisotropy', () => {
    const fit = expectFitReproducesSurface(
      {
        type: 'metric',
        joint: {model: 'exponential', nugget: 0.2, sill: 2, range: 7},
        anisotropy: 0.4
      },
      'pairs'
    );
    if (fit.model.type !== 'metric') throw new Error('type');
    expect(fit.model.joint.range).toBeCloseTo(7, 1);
    expect(fit.model.anisotropy).toBeCloseTo(0.4, 2);
    expect(fit.atBoundary).toBe(false);
  });

  it('fits a sum-metric surface to the bins', () => {
    expectFitReproducesSurface(
      {
        type: 'sum-metric',
        spatial: {model: 'exponential', nugget: 0.1, sill: 0.6, range: 3},
        temporal: {model: 'exponential', nugget: 0.05, sill: 0.4, range: 12},
        joint: {model: 'exponential', nugget: 0.02, sill: 1, range: 6},
        anisotropy: 0.3
      },
      'pairs'
    );
  });

  it('recovers the Gneiting interaction', () => {
    const fit = expectFitReproducesSurface(
      {
        type: 'gneiting',
        nugget: 0.1,
        sill: 2,
        spatialRange: 5,
        temporalRange: 10,
        alpha: 0.5,
        gamma: 0.5,
        beta: 0.6,
        tau: 1
      },
      'none'
    );
    if (fit.model.type !== 'gneiting') throw new Error('type');
    expect(fit.model.beta).toBeCloseTo(0.6, 1);
    expect(fit.model.spatialRange).toBeCloseTo(5, 0);
  });

  it('accepts Matern structures and the Cressie weighting', () => {
    const model: SpaceTimeVariogramModel = {
      type: 'metric',
      joint: {model: 'matern-3/2', nugget: 0.1, sill: 1.5, range: 6},
      anisotropy: 0.5
    };
    const fit = fitSpaceTimeVariogramModel(createSurface(model), {
      type: 'metric',
      structure: 'matern-3/2',
      weighting: 'cressie'
    });
    if (fit.model.type !== 'metric') throw new Error('type');
    expect(fit.model.joint.model).toBe('matern-3/2');
    expect(fit.model.joint.range).toBeCloseTo(6, 1);
  });

  it('ignores the zero-zero bin and throws on too few usable bins', () => {
    expect(() =>
      fitSpaceTimeVariogramModel(
        {distances: [0, 1], timeLags: [0, 1], semivariances: [1, 1], pairCounts: [10, 0]},
        {type: 'metric'}
      )
    ).toThrow('two non-empty bins');
  });
});

describe('Matern in the variogram models', () => {
  it('shape reaches 95 percent at the effective range', () => {
    expect(getVariogramModelShape('matern-3/2', 1)).toBeCloseTo(1 - Math.exp(-3), 12);
    expect(getVariogramModelShape('matern-5/2', 1)).toBeCloseTo(1 - Math.exp(-3), 12);
  });

  for (const model of ['matern-3/2', 'matern-5/2'] as const) {
    it(`fitVariogramModel recovers ${model}`, () => {
      const lags = Array.from({length: 20}, (_, index) => (index + 1) * 0.5);
      const truth = {model, nugget: 0.05, sill: 2, range: 4};
      const fit = fitVariogramModel(
        {
          distances: lags,
          semivariances: lags.map(lag => evaluateVariogramModel(truth, lag)),
          pairCounts: lags.map(() => 100)
        },
        {model, weighting: 'pairs'}
      );
      expect(fit.nugget).toBeCloseTo(0.05, 4);
      expect(fit.sill).toBeCloseTo(2, 4);
      expect(fit.range).toBeCloseTo(4, 4);
    });
  }
});
