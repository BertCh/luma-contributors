// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUSpaceTimeHawkesOperation} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation';
import {runPoisonedOperation} from '../point-process/output-oracle-harness';
import {getSquareInput, evaluateSquare, expectSquareFit} from './hawkes-finite-window-w5-fixture';

it('UPT-01 device finite-window objective increases on the first step and has a stationary boundary fit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = getSquareInput();
  for (const maximumIterations of [1, 1000]) {
    const result = await runPoisonedOperation(
      device,
      new GPUSpaceTimeHawkesOperation({
        rowCount: 4,
        maximumIterations,
        maximumParents: 4,
        positions: 'positions',
        times: 'times',
        parameters: 'parameters',
        output: {
          estimates: 'estimates',
          logLikelihood: 'entering',
          fittedLogLikelihood: 'fitted',
          status: {converged: 'converged', maximumIterations: 'budget'}
        }
      }),
      {
        positions: {format: 'float32x2', values: input.positions},
        times: {format: 'float32', values: input.times},
        parameters: {format: 'float32', values: input.parameters}
      },
      ['estimates', 'entering', 'fitted', 'converged', 'budget']
    );
    expect(Array.from(result['budget'])).toEqual([maximumIterations]);
    if (maximumIterations === 1) {
      const entering = evaluateSquare(input.parameters.subarray(5));
      expect(Number.isFinite(result['fitted'][0])).toBe(true);
      expect(Math.abs(result['entering'][0] - entering)).toBeLessThan(2e-5);
      expect(result['fitted'][0]).toBeGreaterThanOrEqual(entering);
    } else expectSquareFit(result['estimates'], result['fitted'][0], result['converged'][0]);
  }
}, 120000);

it('DEF-13 device omission and explicit 100 publish the same fit, count and resolved budget', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results = [];
  for (const maximumIterations of [undefined, 100])
    results.push(
      await runPoisonedOperation(
        device,
        new GPUSpaceTimeHawkesOperation({
          rowCount: 8,
          maximumIterations,
          positions: 'positions',
          times: 'times',
          parameters: 'parameters',
          output: {
            estimates: 'estimates',
            fittedLogLikelihood: 'fitted',
            status: {
              iterationCount: 'iterations',
              maximumIterations: 'budget',
              converged: 'converged',
              iterationLimitReached: 'limit'
            }
          }
        }),
        {
          positions: {
            format: 'float32x2',
            values: Float32Array.of(2, 4, 2, 5, 3, 5, 4, 3, 1, 3, 0, 1, 4, 3, 7, 4)
          },
          times: {format: 'float32', values: Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7)},
          parameters: {
            format: 'float32',
            values: Float32Array.of(30, 8, 8, 15, 1e-5, 0.05, 0.5, 1, 2)
          }
        },
        ['estimates', 'fitted', 'iterations', 'budget', 'converged', 'limit']
      )
    );
  expect(results[0]).toEqual(results[1]);
  expect(Array.from(results[0]['budget'])).toEqual([100]);
  for (const value of results[0]['estimates']) expect(Number.isFinite(value)).toBe(true);
}, 120000);
