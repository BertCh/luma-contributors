// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpaceTimeHawkesOperation} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {runHawkesGPU} from './stm08-hawkes-harness';

for (const maximumIterations of [1, 3]) {
  it(`Hawkes contributor and operation use the hand-derived Poisson stopping decision at budget ${maximumIterations}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Both events occur at t=2, so neither has a strictly earlier parent. The first update
    // changes background from 1 to 2/(1*10)=0.2 (relative change 0.8 > 1e-4) and sets the
    // branching ratio to its floor. The second update is identical: relative change 0.
    // Thus budget 1 must leave work, while budget 3 finishes after exactly two updates.
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 1)},
        times: {format: 'float32', values: Float32Array.of(2, 2)},
        parameters: {format: 'float32', values: Float32Array.of(1, 10, 10, 10, 1e-4, 1, 0.5, 1, 1)}
      },
      operations: [
        new GPUSpaceTimeHawkesOperation({
          rowCount: 2,
          maximumIterations,
          maximumParents: 4,
          positions: 'positions',
          times: 'times',
          parameters: 'parameters',
          output: {
            estimates: 'estimates',
            status: {
              converged: 'converged',
              iterationLimitReached: 'limited',
              iterationCount: 'iterations'
            }
          }
        })
      ],
      outputs: ['converged', 'limited', 'iterations']
    });
    expect(Array.from(result['converged'])).toEqual([maximumIterations === 1 ? 0 : 1]);
    expect(Array.from(result['limited'])).toEqual([maximumIterations === 1 ? 1 : 0]);
    expect(Array.from(result['iterations'])).toEqual([maximumIterations === 1 ? 1 : 2]);
    const contributor = await runHawkesGPU(
      device,
      [
        {x: 0, y: 0, time: 2},
        {x: 1, y: 1, time: 2}
      ],
      {area: 1, duration: 10, maximumLag: 10, maximumDistance: 10, maximumParents: 4},
      {background: 1, branchingRatio: 0.5, decayRate: 1, spatialStandardDeviation: 1},
      {maximumIterations, tolerance: 1e-4}
    );
    expect(contributor.converged).toBe(maximumIterations === 1 ? 0 : 1);
    expect(contributor.iterationLimitReached).toBe(maximumIterations === 1 ? 1 : 0);
    expect(contributor.iterationCount).toBe(maximumIterations === 1 ? 1 : 2);
  });
}
