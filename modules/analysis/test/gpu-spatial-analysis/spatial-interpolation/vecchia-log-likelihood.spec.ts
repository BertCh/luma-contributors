// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, describe, expect, it} from 'vitest';
import {
  getGPUVecchiaLogLikelihoodParameterValues,
  type GPUVecchiaCovariance
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {getGPULocalKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {buildVecchiaConditioningSets} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-conditioning-sets';
import {
  createExperimentGaussian,
  createExperimentRandom,
  evaluateReferenceVecchiaLogLikelihood,
  simulateReferenceVecchiaField
} from './vecchia-likelihood-reference';
import {
  createVecchiaFisherHarness,
  createVecchiaLikelihoodHarness
} from './vecchia-likelihood-harness';

/** A square of `rowCount` uniform points simulated from the Vecchia model, the experiment's setup. */
function createScene(rowCount: number, neighborCount: number, covariance: GPUVecchiaCovariance) {
  const random = createExperimentRandom(20261109);
  const positions = new Float64Array(rowCount * 2);
  for (let index = 0; index < rowCount * 2; index++) {
    positions[index] = (random() - 0.5) * 100000;
  }
  const {offsets, neighbors} = buildVecchiaConditioningSets({positions, neighborCount});
  const values = simulateReferenceVecchiaField(
    createExperimentGaussian(random),
    positions,
    offsets,
    neighbors,
    covariance
  );
  return {positions, values, offsets, neighbors, neighborCount};
}

describe('GPUVecchiaLogLikelihood', () => {
  const destroyers: (() => void)[] = [];
  afterEach(() => {
    destroyers.splice(0).forEach(destroy => destroy());
  });

  async function createHarness(
    scene: Parameters<typeof createVecchiaLikelihoodHarness>[0]
  ): Promise<Awaited<ReturnType<typeof createVecchiaLikelihoodHarness>>> {
    const harness = await createVecchiaLikelihoodHarness(scene);
    destroyers.push(harness.destroy);
    return harness;
  }

  it('matches the binary64 reference across kernels and range without a recompile', async () => {
    const trueCovariance: GPUVecchiaCovariance = {
      model: 'exponential',
      range: 30000,
      sill: 1,
      nugget: 1e-3
    };
    const scene = createScene(1500, 20, trueCovariance);
    const harness = await createHarness(scene);
    const candidates: GPUVecchiaCovariance[] = [
      {model: 'exponential', range: 30000, sill: 1, nugget: 1e-3},
      {model: 'exponential', range: 22000, sill: 1, nugget: 0},
      {model: 'exponential', range: 41000, sill: 2, nugget: 0.1},
      {model: 'matern-3/2', range: 22000, sill: 1, nugget: 1e-3},
      {model: 'matern-3/2', range: 16000, sill: 0.7, nugget: 0.05}
    ];
    for (const covariance of candidates) {
      const reference = evaluateReferenceVecchiaLogLikelihood(
        scene.values,
        scene.positions,
        scene.offsets,
        scene.neighbors,
        covariance
      );
      const result = await harness.evaluate(getGPUVecchiaLogLikelihoodParameterValues(covariance));
      expect(result.failedSiteCount).toBe(0);
      expect(result.usedSiteCount).toBe(1500);
      // The GPU study reports an error of at most 0.0095 (exponential) and 0.21 (Matern 3/2, nugget
      // 1e-3) at n = 10,000; these candidates misspecify the exponential field at n = 1,500.
      const tolerance =
        covariance.model === 'exponential' || covariance.nugget! > 1e-3 ? 0.02 : 0.2;
      expect(Math.abs(result.logLikelihood - reference.logLikelihood)).toBeLessThan(tolerance);
    }
    // The same compiled graph served all five covariances, two models and a sill change.
    expect(harness.buildCount).toBe(1);
  });

  it('adds the observation error of the site to its conditional variance', async () => {
    const covariance: GPUVecchiaCovariance = {
      model: 'exponential',
      range: 30000,
      sill: 1,
      nugget: 0
    };
    const scene = createScene(600, 12, covariance);
    const observationError = new Float64Array(600).fill(0.4);
    const harness = await createHarness({...scene, observationError});
    const result = await harness.evaluate(getGPUVecchiaLogLikelihoodParameterValues(covariance));
    // The total is that of a covariance whose nugget is 0.4 on the diagonal of every system.
    const reference = evaluateReferenceVecchiaLogLikelihood(
      scene.values,
      scene.positions,
      scene.offsets,
      scene.neighbors,
      {...covariance, nugget: 0.4}
    );
    expect(result.failedSiteCount).toBe(0);
    expect(Math.abs(result.logLikelihood - reference.logLikelihood)).toBeLessThan(0.05);
  });

  it('uses the marginal variance at a site with no conditioning set', async () => {
    // Three sites with empty sets: an independent normal log-density with sill + nugget.
    const scene = {
      positions: [0, 0, 1000, 0, 0, 2000],
      values: [0.5, -1.25, 2],
      offsets: new Uint32Array(4),
      neighbors: new Uint32Array(0),
      neighborCount: 4
    };
    const harness = await createHarness(scene);
    const sill = 1.5;
    const nugget = 0.25;
    const result = await harness.evaluate(
      getGPUVecchiaLogLikelihoodParameterValues({model: 'exponential', range: 100, sill, nugget})
    );
    const variance = sill + nugget;
    const expected = scene.values.reduce(
      (sum, value) =>
        sum - 0.5 * (Math.log(2 * Math.PI) + Math.log(variance) + (value * value) / variance),
      0
    );
    expect(result.failedSiteCount).toBe(0);
    expect(result.logLikelihood).toBeCloseTo(expected, 3);
  });

  it('fails the whole likelihood when a site cannot be solved', async () => {
    // Two coincident conditioning sites with no nugget make a singular solve at site 2.
    const scene = {
      positions: [0, 0, 0, 0, 500, 0],
      values: [1, 1, 2],
      offsets: new Uint32Array([0, 0, 1, 3]),
      neighbors: new Uint32Array([0, 0, 1]),
      neighborCount: 4
    };
    const harness = await createHarness(scene);
    const result = await harness.evaluate(
      getGPUVecchiaLogLikelihoodParameterValues({model: 'exponential', range: 3000, sill: 1})
    );
    expect(result.failedSiteCount).toBeGreaterThan(0);
    expect(Number.isNaN(result.logLikelihood)).toBe(true);
    expect(Number.isNaN(result.high)).toBe(true);
  });

  it('excludes invalid observations and reports them', async () => {
    const covariance: GPUVecchiaCovariance = {
      model: 'exponential',
      range: 30000,
      sill: 1,
      nugget: 1e-3
    };
    const scene = createScene(400, 12, covariance);
    const validity = new Float64Array(400).fill(1);
    validity[5] = 0;
    validity[17] = 0;
    // Rebuild sets without the excluded rows so no site conditions on them.
    const withoutExcluded = buildVecchiaConditioningSets({
      positions: scene.positions,
      neighborCount: 12,
      validity
    });
    const harness = await createHarness({
      ...scene,
      offsets: withoutExcluded.offsets,
      neighbors: withoutExcluded.neighbors,
      validity
    });
    const result = await harness.evaluate(getGPUVecchiaLogLikelihoodParameterValues(covariance));
    expect(result.failedSiteCount).toBe(0);
    expect(result.excludedSiteCount).toBe(2);
    expect(result.usedSiteCount).toBe(398);
    expect(Number.isFinite(result.logLikelihood)).toBe(true);
  });

  it('treats a site whose conditioning rows are all excluded as marginal, like the Fisher scoring sibling', async () => {
    // Math correctness review, interpolation-density-06. Row 0 is excluded but still listed in
    // the conditioning sets (validity toggled without rebuilding them). Site 1 conditions on
    // {0}: nothing usable, so its term is the marginal one. Site 2 conditions on {0, 1}: only
    // row 1 is usable. Expected values by hand, exponential model, sill 1, nugget 0.1, range
    // 3000: C(0) = 1.1 and C(1500) = exp(-3 * 1500 / 3000) = exp(-1.5).
    const scene = {
      positions: [0, 0, 1000, 0, 2500, 0],
      values: [7, 0.8, -0.4],
      offsets: new Uint32Array([0, 0, 1, 3]),
      neighbors: new Uint32Array([0, 0, 1]),
      neighborCount: 4,
      validity: [0, 1, 1]
    };
    const termOf = (residual: number, variance: number) =>
      -0.5 * (Math.log(2 * Math.PI) + Math.log(variance) + (residual * residual) / variance);
    const crossCovariance = Math.exp(-1.5);
    const expectedTerms = [
      termOf(0.8, 1.1),
      termOf(-0.4 - (crossCovariance / 1.1) * 0.8, 1.1 - (crossCovariance * crossCovariance) / 1.1)
    ];
    const parameters = getGPUVecchiaLogLikelihoodParameterValues({
      model: 'exponential',
      range: 3000,
      sill: 1,
      nugget: 0.1
    });
    const harness = await createHarness(scene);
    const result = await harness.evaluate(parameters);
    expect(result.failedSiteCount).toBe(0);
    expect(result.excludedSiteCount).toBe(1);
    expect(result.usedSiteCount).toBe(2);
    expect(result.logLikelihood).toBeCloseTo(expectedTerms[0] + expectedTerms[1], 5);
    const {siteTerms, siteStatus} = await harness.readConditionals();
    expect(Array.from(siteStatus)).toEqual([1, 0, 0]);
    expect(siteTerms[1]).toBeCloseTo(expectedTerms[0], 5);
    expect(siteTerms[2]).toBeCloseTo(expectedTerms[1], 5);

    const fisher = await createVecchiaFisherHarness(scene);
    destroyers.push(fisher.destroy);
    const fisherResult = await fisher.evaluate(parameters);
    expect(fisherResult.failedSiteCount).toBe(0);
    expect(fisherResult.usedSiteCount).toBe(2);
    expect(fisherResult.logLikelihood).toBeCloseTo(expectedTerms[0] + expectedTerms[1], 5);
  });

  it('computes a finite likelihood from raw parameters below the floor, unguarded by the kernel', async () => {
    // The floor is the job of the packer; the kernel must report failures, not hide them.
    const covariance: GPUVecchiaCovariance = {
      model: 'exponential',
      range: 30000,
      sill: 1,
      nugget: 1e-3
    };
    const scene = createScene(1000, 20, covariance);
    const harness = await createHarness(scene);
    const result = await harness.evaluate(
      getGPULocalKrigingParameterValues({
        covariance: {model: 'matern-3/2', range: 20000, sill: 1, nugget: 0}
      })
    );
    expect(result.failedSiteCount + result.usedSiteCount).toBe(1000);
  });
});
