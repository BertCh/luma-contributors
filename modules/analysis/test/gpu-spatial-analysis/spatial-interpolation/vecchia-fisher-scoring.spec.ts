// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, describe, expect, it} from 'vitest';
import {buildVecchiaConditioningSets} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-conditioning-sets';
import {fitVecchiaCovarianceFisherScoring} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-fisher-fit';
import {
  getGPUVecchiaLogLikelihoodParameterValues,
  type GPUVecchiaCovariance
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {
  createExperimentGaussian,
  createExperimentRandom,
  evaluateReferenceVecchiaFisher,
  simulateReferenceVecchiaField
} from './vecchia-likelihood-reference';
import {createVecchiaFisherHarness} from './vecchia-likelihood-harness';

function createScene(
  rowCount: number,
  neighborCount: number,
  seed: number,
  truth: GPUVecchiaCovariance
) {
  const random = createExperimentRandom(seed);
  const positions = Float64Array.from({length: rowCount * 2}, () => (random() - 0.5) * 100000);
  const {offsets, neighbors} = buildVecchiaConditioningSets({positions, neighborCount});
  const values = simulateReferenceVecchiaField(
    createExperimentGaussian(random),
    positions,
    offsets,
    neighbors,
    truth
  );
  return {positions, values, offsets, neighbors, neighborCount};
}

describe('GPUVecchiaFisherScoring', () => {
  const destroyers: (() => void)[] = [];
  afterEach(() => destroyers.splice(0).forEach(destroy => destroy()));

  const truths: GPUVecchiaCovariance[] = [
    {model: 'exponential', range: 30000, sill: 1.5, nugget: 0.05},
    {model: 'matern-3/2', range: 20000, sill: 1, nugget: 0.02}
  ];

  it('matches the binary64 score and information without a recompile', async () => {
    const scene = createScene(1500, 20, 41, truths[0]);
    const harness = await createVecchiaFisherHarness(scene);
    destroyers.push(harness.destroy);
    const candidates: GPUVecchiaCovariance[] = [
      {model: 'exponential', range: 42000, sill: 1, nugget: 0.1},
      {model: 'exponential', range: 21000, sill: 2, nugget: 0.01},
      {model: 'matern-3/2', range: 24000, sill: 1.2, nugget: 0.05}
    ];
    for (const covariance of candidates) {
      const reference = evaluateReferenceVecchiaFisher(
        scene.values,
        scene.positions,
        scene.offsets,
        scene.neighbors,
        covariance
      );
      const result = await harness.evaluate(getGPUVecchiaLogLikelihoodParameterValues(covariance));
      expect(result.failedSiteCount).toBe(0);
      expect(Math.abs(result.logLikelihood - reference.logLikelihood)).toBeLessThan(0.05);
      for (let entry = 0; entry < 3; entry++) {
        expect(Math.abs(result.score[entry] - reference.score[entry])).toBeLessThan(
          1e-3 *
            (Math.abs(reference.score[entry]) +
              reference.information[entry === 0 ? 0 : entry === 1 ? 3 : 5])
        );
      }
      for (let entry = 0; entry < 6; entry++) {
        expect(result.information[entry]).toBeCloseTo(reference.information[entry], -1);
        expect(Math.abs(result.information[entry] - reference.information[entry])).toBeLessThan(
          2e-3 * (Math.abs(reference.information[entry]) + 1)
        );
      }
    }
    expect(harness.buildCount).toBe(1);
  });

  it('marks a singular system as a failed evaluation', async () => {
    const harness = await createVecchiaFisherHarness({
      positions: [0, 0, 0, 0, 500, 0],
      values: [1, 1, 2],
      offsets: new Uint32Array([0, 0, 1, 3]),
      neighbors: new Uint32Array([0, 0, 1]),
      neighborCount: 4
    });
    destroyers.push(harness.destroy);
    const result = await harness.evaluate(
      getGPUVecchiaLogLikelihoodParameterValues({model: 'exponential', range: 3000, sill: 1})
    );
    expect(result.failedSiteCount).toBeGreaterThan(0);
    expect(Number.isNaN(result.logLikelihood)).toBe(true);
  });

  it('fits float32 evaluations to the binary64 estimate across seeds and kernels', {
    timeout: 600000
  }, async () => {
    const summaries: string[] = [];
    for (const truth of truths) {
      for (const seed of [11, 12, 13]) {
        const scene = createScene(2500, 20, seed, truth);
        const harness = await createVecchiaFisherHarness(scene);
        destroyers.push(harness.destroy);
        const initial: GPUVecchiaCovariance = {
          model: truth.model,
          range: truth.range * 1.8,
          sill: truth.sill * 0.5,
          nugget: truth.nugget! * 4
        };
        const gpuFit = await fitVecchiaCovarianceFisherScoring({
          initial,
          evaluate: covariance =>
            harness.evaluate(getGPUVecchiaLogLikelihoodParameterValues(covariance))
        });
        const referenceFit = await fitVecchiaCovarianceFisherScoring({
          initial,
          logLikelihoodNoise: 1e-9,
          evaluate: covariance => ({
            ...evaluateReferenceVecchiaFisher(
              scene.values,
              scene.positions,
              scene.offsets,
              scene.neighbors,
              covariance
            ),
            failedSiteCount: 0
          })
        });
        summaries.push(
          `${truth.model} seed=${seed} iters gpu=${gpuFit.iterationCount} ref=${referenceFit.iterationCount} converged=${gpuFit.converged}/${referenceFit.converged} gpu=[${gpuFit.covariance.sill.toPrecision(5)},${gpuFit.covariance.range.toFixed(0)},${gpuFit.covariance.nugget!.toPrecision(4)}] ref=[${referenceFit.covariance.sill.toPrecision(5)},${referenceFit.covariance.range.toFixed(0)},${referenceFit.covariance.nugget!.toPrecision(4)}] dLL=${(gpuFit.logLikelihood - referenceFit.logLikelihood).toFixed(4)}`
        );
        expect(gpuFit.failed).toBe(false);
        expect(gpuFit.converged).toBe(true);
        expect(referenceFit.converged).toBe(true);
        expect(harness.buildCount).toBe(1);
        expect(Math.abs(gpuFit.logLikelihood - referenceFit.logLikelihood)).toBeLessThan(0.1);
        expect(
          Math.abs(Math.log(gpuFit.covariance.sill / referenceFit.covariance.sill))
        ).toBeLessThan(0.02);
        expect(
          Math.abs(Math.log(gpuFit.covariance.range / referenceFit.covariance.range))
        ).toBeLessThan(0.02);
        expect(
          Math.abs(Math.log(gpuFit.covariance.nugget! / referenceFit.covariance.nugget!))
        ).toBeLessThan(0.05);
      }
    }
    console.log(`VECCHIA_FISHER_FITS\n${summaries.join('\n')}\nVECCHIA_FISHER_FITS_END`);
  });
  it('stops at the nugget floor and at a vanishing exponential nugget like binary64', {
    timeout: 600000
  }, async () => {
    const cases: {truth: GPUVecchiaCovariance; nuggetStart: number; expectFloor: boolean}[] = [
      {
        truth: {model: 'matern-3/2', range: 20000, sill: 1, nugget: 1e-5},
        nuggetStart: 0.02,
        expectFloor: true
      },
      {
        truth: {model: 'exponential', range: 30000, sill: 1, nugget: 1e-6},
        nuggetStart: 0.05,
        expectFloor: false
      }
    ];
    for (const {truth, nuggetStart, expectFloor} of cases) {
      const scene = createScene(2500, 20, 31, truth);
      const harness = await createVecchiaFisherHarness(scene);
      destroyers.push(harness.destroy);
      const initial: GPUVecchiaCovariance = {
        model: truth.model,
        range: truth.range * 1.6,
        sill: truth.sill * 0.6,
        nugget: nuggetStart
      };
      const gpuFit = await fitVecchiaCovarianceFisherScoring({
        initial,
        evaluate: covariance =>
          harness.evaluate(getGPUVecchiaLogLikelihoodParameterValues(covariance))
      });
      const referenceFit = await fitVecchiaCovarianceFisherScoring({
        initial,
        logLikelihoodNoise: 1e-9,
        evaluate: covariance => ({
          ...evaluateReferenceVecchiaFisher(
            scene.values,
            scene.positions,
            scene.offsets,
            scene.neighbors,
            covariance
          ),
          failedSiteCount: 0
        })
      });
      console.log(
        `VECCHIA_FISHER_BOUND ${truth.model} gpu iters=${gpuFit.iterationCount} converged=${gpuFit.converged} floor=${gpuFit.atNuggetFloor} [${gpuFit.covariance.sill},${gpuFit.covariance.range},${gpuFit.covariance.nugget}] ll=${gpuFit.logLikelihood}; ref iters=${referenceFit.iterationCount} converged=${referenceFit.converged} floor=${referenceFit.atNuggetFloor} [${referenceFit.covariance.sill},${referenceFit.covariance.range},${referenceFit.covariance.nugget}] ll=${referenceFit.logLikelihood}`
      );
      expect(gpuFit.failed).toBe(false);
      expect(gpuFit.atNuggetFloor).toBe(expectFloor);
      expect(referenceFit.atNuggetFloor).toBe(expectFloor);
      if (expectFloor) {
        // At the floor the expected information is a poor model of the likelihood near a vanishing
        // nugget: Fisher scoring zigzags in binary64 as well, so `converged` is not expected here
        // and an honest limit is reported. Binary64 and float32 still agree to a few units.
        expect(Math.abs(gpuFit.logLikelihood - referenceFit.logLikelihood)).toBeLessThan(3);
      } else {
        expect(gpuFit.converged).toBe(true);
        expect(
          Math.abs(Math.log(gpuFit.covariance.range / referenceFit.covariance.range))
        ).toBeLessThan(0.02);
        expect(Math.abs(gpuFit.logLikelihood - referenceFit.logLikelihood)).toBeLessThan(0.1);
      }
    }
  });
});
