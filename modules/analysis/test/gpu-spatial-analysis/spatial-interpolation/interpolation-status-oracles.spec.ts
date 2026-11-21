// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {getGPUGaussianRandomFieldParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gaussian-random-field-parameters';
import {createFieldRun} from './gaussian-random-field-test-utils';
import fixture from './fixtures/gaussian-random-field.json';
import {GPUGaussianRandomField} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-gaussian-random-field';
import {GraphHarness} from '../trajectory-staypoints/movement-spec-utils';
import {readUint32} from '../../utils/gpu-contributor-test-utils';

it('Gaussian point validity overwrites the active row and both poisoned padding rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'Gaussian-validity-status-oracle');
  const validity = harness.output('validity', 'uint32', 3);
  validity.buffer.write(Uint32Array.from([37, 41, 43]));
  harness.graph.add(
    new GPUGaussianRandomField({
      positions: harness.input('positions', Float32Array.from([0, 0]), 'float32x2', 1),
      parameters: harness.input(
        'parameters',
        getGPUGaussianRandomFieldParameterValues({
          variogram: {model: 'gaussian', nugget: 1, sill: 0, range: 1},
          seed: 7
        }),
        'float32',
        16
      ),
      method: 'spectral',
      modeCount: 1,
      output: {
        values: harness.output('values', 'float32', 3).view,
        validity: validity.view,
        status: {approximation: harness.output('approximation', 'uint32', 1).view},
        clippedEigenvalueMass: harness.output('mass', 'float32', 1).view
      }
    })
  );
  // A finite point with a valid unit-nugget covariance is usable. The other two
  // output rows have no domain point and must be invalid, regardless of prior bytes.
  await harness.run(async () => expect(await readUint32(validity.buffer, 3)).toEqual([1, 0, 0]));
});

it('Gaussian field approximation follows independent spectrum and clears after clipping stops', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // NumPy fixture has strictly positive clipped mass 0.027091352676814562
  // for this FFT embedding. Approximation must be 1 independently of the
  // GPU's mass output. A pure unit nugget is a delta covariance whose FFT
  // eigenvalues are all 1, so no eigenvalue is clipped and approximation=0.
  const scenario = fixture.scenarios.find(scenario => scenario.name === 'fft-clipped-long-range')!;
  expect(scenario.clippedEigenvalueMass).toBe(0.027091352676814562);
  const run = createFieldRun(device, scenario);
  try {
    await run.execute();
    expect((await run.status()).approximation).toBe(1);
    run.parameterBuffer.write(
      getGPUGaussianRandomFieldParameterValues({
        ...run.settings,
        variogram: {model: 'exponential', nugget: 1, sill: 0, range: 1}
      })
    );
    await run.execute();
    expect((await run.status()).approximation).toBe(0);
    // Replay must restore the nonzero flag on the same graph.
    run.parameterBuffer.write(getGPUGaussianRandomFieldParameterValues(run.settings));
    await run.execute();
    expect((await run.status()).approximation).toBe(1);
  } finally {
    run.destroy();
  }
});
