// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUAdamStep} from '../../../src/analysis-core/kernels/gpu-adam-step';
import {getGPUAdamStepParameterValues} from '../../../src/analysis-core/kernels/adam-step-parameters';
import {makeSpatialFixture, readIntegers} from '../../spatial-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const stable of [false, true]) {
  it(`Adam reports a one-update budget and clears it only for an independently stable loss: ${stable}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const gradients = fixture.view('float32', new Float32Array([stable ? 0 : 1]));
      const loss = fixture.view('float32', new Float32Array([1]));
      const parameters = fixture.view(
        'float32',
        getGPUAdamStepParameterValues({learningRate: 0.25, beta1: 0, beta2: 0, tolerance: 0.01})
      );
      const weights = fixture.view('float32', new Float32Array([1]));
      const firstMoment = fixture.view('float32', new Float32Array([0]));
      const secondMoment = fixture.view('float32', new Float32Array([0]));
      const lossHistory = fixture.view('float32', new Float32Array([-99, -99]));
      // Adam status is persistent state and must start at zero; poisoning it
      // would mean an already converged/invalid fit. Exercise a real 1 -> 0
      // transition instead of relying on fresh zero storage.
      const iterationCount = fixture.view('uint32', new Uint32Array([0]));
      const converged = fixture.view('uint32', new Uint32Array([0]));
      const iterationLimitReached = fixture.view('uint32', new Uint32Array([0]));
      const invalidCount = fixture.view('uint32', new Uint32Array([0]));
      fixture.graph.add(
        new GPUAdamStep({
          gradients: gradients.view,
          loss: loss.view,
          parameters: parameters.view,
          maximumIterations: 1,
          output: {
            weights: weights.view,
            firstMoment: firstMoment.view,
            secondMoment: secondMoment.view,
            lossHistory: lossHistory.view,
            status: {
              iterationCount: iterationCount.view,
              converged: converged.view,
              iterationLimitReached: iterationLimitReached.view,
              invalidCount: invalidCount.view
            }
          }
        })
      );
      const readStatus = async () => [
        ...(await readIntegers(iterationCount.buffer)),
        ...(await readIntegers(converged.buffer)),
        ...(await readIntegers(iterationLimitReached.buffer)),
        ...(await readIntegers(invalidCount.buffer))
      ];
      await fixture.run();
      expect(await readStatus()).toEqual([1, 0, 1, 0]);
      // Stable: constant loss 1 and zero gradient/moments give an exact fixed
      // point. Unstable: L(w)=w, derivative 1, w moves from 1 to ~0.75 and
      // work remains (derivative still 1); loss change ~0.25 exceeds 0.01.
      loss.buffer.write(new Float32Array([stable ? 1 : 0.75]));
      await fixture.run();
      expect(await readStatus()).toEqual(stable ? [1, 1, 0, 0] : [1, 0, 1, 0]);
      await fixture.run();
      expect(await readStatus()).toEqual(stable ? [1, 1, 0, 0] : [1, 0, 1, 0]);
    } finally {
      fixture.destroy();
    }
  });
}
