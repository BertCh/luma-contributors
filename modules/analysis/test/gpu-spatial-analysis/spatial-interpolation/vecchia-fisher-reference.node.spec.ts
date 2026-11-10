// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {buildVecchiaConditioningSets} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-conditioning-sets';
import {fitVecchiaCovarianceFisherScoring} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-fisher-fit';
import {
  createExperimentGaussian,
  createExperimentRandom,
  evaluateReferenceVecchiaFisher,
  evaluateReferenceVecchiaLogLikelihood,
  simulateReferenceVecchiaField,
  type ReferenceVecchiaCovariance,
  type ReferenceVecchiaKernel
} from './vecchia-likelihood-reference';

function createScene(rowCount: number, neighborCount: number, seed: number) {
  const random = createExperimentRandom(seed);
  const positions = Float64Array.from({length: rowCount * 2}, () => (random() - 0.5) * 100);
  const {offsets, neighbors} = buildVecchiaConditioningSets({positions, neighborCount});
  return {positions, offsets, neighbors, gaussian: createExperimentGaussian(random)};
}

const CASES: ReferenceVecchiaCovariance[] = [
  {model: 'exponential', range: 30, sill: 2, nugget: 0.05},
  {model: 'matern-3/2', range: 18, sill: 0.6, nugget: 0.03}
];

describe('Vecchia score and expected information reference', () => {
  for (const covariance of CASES) {
    it(`score matches finite differences of the likelihood: ${covariance.model}`, () => {
      const scene = createScene(120, 8, 5);
      const values = simulateReferenceVecchiaField(
        scene.gaussian,
        scene.positions,
        scene.offsets,
        scene.neighbors,
        covariance
      );
      const fisher = evaluateReferenceVecchiaFisher(
        values,
        scene.positions,
        scene.offsets,
        scene.neighbors,
        covariance
      );
      const ll = (log: number[]) =>
        evaluateReferenceVecchiaLogLikelihood(
          values,
          scene.positions,
          scene.offsets,
          scene.neighbors,
          {
            model: covariance.model as ReferenceVecchiaKernel,
            sill: Math.exp(log[0]),
            range: Math.exp(log[1]),
            nugget: Math.exp(log[2])
          }
        ).logLikelihood;
      const theta = [
        Math.log(covariance.sill),
        Math.log(covariance.range),
        Math.log(covariance.nugget!)
      ];
      for (let parameter = 0; parameter < 3; parameter++) {
        const step = 1e-5;
        const up = theta.slice();
        const down = theta.slice();
        up[parameter] += step;
        down[parameter] -= step;
        const difference = (ll(up) - ll(down)) / (2 * step);
        expect(fisher.score[parameter]).toBeCloseTo(difference, 4);
      }
    });

    it(`information equals the expected score outer product: ${covariance.model}`, () => {
      const scene = createScene(60, 6, 11);
      const reference = evaluateReferenceVecchiaFisher(
        new Float64Array(60),
        scene.positions,
        scene.offsets,
        scene.neighbors,
        covariance
      );
      const outer = [0, 0, 0, 0, 0, 0];
      const runs = 4000;
      for (let run = 0; run < runs; run++) {
        const values = simulateReferenceVecchiaField(
          scene.gaussian,
          scene.positions,
          scene.offsets,
          scene.neighbors,
          covariance
        );
        const {score} = evaluateReferenceVecchiaFisher(
          values,
          scene.positions,
          scene.offsets,
          scene.neighbors,
          covariance
        );
        let cell = 0;
        for (let first = 0; first < 3; first++) {
          for (let second = first; second < 3; second++) {
            outer[cell++] += (score[first] * score[second]) / runs;
          }
        }
      }
      // E[s s'] = I for the true parameters; the Monte Carlo error is about 1/sqrt(runs).
      const diagonals = [0, 3, 5];
      for (let cell = 0; cell < 6; cell++) {
        const scale = Math.sqrt(
          reference.information[diagonals[Math.min(cell, 2)]] * reference.information[5]
        );
        expect(Math.abs(outer[cell] - reference.information[cell])).toBeLessThan(
          0.15 * Math.max(scale, Math.abs(reference.information[cell]))
        );
      }
    });
  }

  it('fits a spatial exponential field with the shared driver in binary64', async () => {
    const scene = createScene(1500, 15, 91);
    const truth: ReferenceVecchiaCovariance = {
      model: 'exponential',
      range: 30,
      sill: 1.2,
      nugget: 0.05
    };
    const values = simulateReferenceVecchiaField(
      scene.gaussian,
      scene.positions,
      scene.offsets,
      scene.neighbors,
      truth
    );
    const fit = await fitVecchiaCovarianceFisherScoring({
      initial: {model: 'exponential', sill: 0.5, range: 60, nugget: 0.2},
      evaluate: covariance => ({
        ...evaluateReferenceVecchiaFisher(
          values,
          scene.positions,
          scene.offsets,
          scene.neighbors,
          covariance
        ),
        failedSiteCount: 0
      })
    });
    expect(fit.converged).toBe(true);
    expect(fit.iterationCount).toBeLessThanOrEqual(15);
    // Estimates near the truth, within sampling error at n = 1500.
    expect(Math.abs(Math.log(fit.covariance.range / truth.range))).toBeLessThan(0.4);
    expect(Math.abs(Math.log(fit.covariance.sill / truth.sill))).toBeLessThan(0.4);
  });
});
