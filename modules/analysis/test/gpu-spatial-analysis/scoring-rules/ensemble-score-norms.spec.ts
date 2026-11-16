// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {runEnsembleScores} from './scoring-rules-harness';

it('energy distances preserve a finite large 3-4-5 norm and the independent CRPS closed form', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runEnsembleScores(
    device,
    {memberCount: 2, dimension: 2, ensemble: [3e20, 4e20, 0, 0], observations: [0, 0]},
    {crps: true, energyScore: true}
  );
  // A two-point distribution with its observation at one endpoint has score distance/4.
  expect(Math.abs(result.energyScore! / 1.25e20 - 1)).toBeLessThan(1e-5);
  expect(Math.abs(result.crps![0] / 7.5e19 - 1)).toBeLessThan(1e-5);
  expect(Math.abs(result.crps![1] / 1e20 - 1)).toBeLessThan(1e-5);
});

for (const [name, dimension, ensemble, observations] of [
  ['equal huge components', 2, [3e38, 3e38, 0, 0], [0, 0]],
  ['unequal huge components', 2, [3e38, 1.5e38, 0, 0], [0, 0]],
  ['reversed huge components', 2, [1.5e38, 3e38, 0, 0], [0, 0]],
  ['opposite near-maximum values', 1, [3e38, -3e38], [-3e38]],
  ['three huge components', 3, [3e38, -2e38, 1e38, 0, 0, 0], [0, 0, 0]],
  ['large dynamic range', 3, [3e38, 1e20, -2e38, 0, 0, 0], [0, 0, 0]]
] as const) {
  it(`normalizes distances before overflow: ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const uploaded = Float32Array.from(ensemble);
    // Observation at the second endpoint: the empirical score is distance / 4, in float64.
    const differences = Array.from(
      {length: dimension},
      (_, site) => uploaded[site] - uploaded[dimension + site]
    );
    const expectedEnergy = Math.hypot(...differences) / 4;
    const result = await runEnsembleScores(
      device,
      {
        memberCount: 2,
        dimension,
        ensemble: Array.from(ensemble),
        observations: Array.from(observations)
      },
      {crps: true, energyScore: true}
    );
    expect(Math.abs(result.energyScore! / expectedEnergy - 1)).toBeLessThan(2e-5);
    for (let site = 0; site < dimension; site++) {
      const expected = Math.abs(differences[site]) / 4;
      expect(Math.abs(result.crps![site] / expected - 1)).toBeLessThan(2e-5);
    }
  });
}
