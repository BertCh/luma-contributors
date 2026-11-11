// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURestrictedRegimeRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation';
import {GPURestrictedRegimeRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './restricted-regime-regression-operation-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

it('preserves small response variation next to a different regime at 1e20', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const largeOrigin = Math.fround(1e20);
  const names = ['coefficients', 'covariance', 'fitted', 'chow', 'status'];
  const outputs: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
  for (const Operation of [
    GPURestrictedRegimeRegressionOperation,
    CPURestrictedRegimeRegressionOperation
  ]) {
    outputs.push(
      await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: Float32Array.of(-1, 1, -1, 1, -1, 1, -1, 1)},
          regimes: {format: 'uint32', values: Uint32Array.of(0, 0, 0, 0, 1, 1, 1, 1)},
          response: {
            format: 'float32',
            values: Float32Array.of(largeOrigin, largeOrigin, largeOrigin, largeOrigin, 0, 2, 0, 2)
          }
        },
        operations: [
          new Operation({
            rowCount: 8,
            predictorCount: 1,
            regimeCount: 2,
            common: [false, true],
            predictors: 'predictors',
            regimes: 'regimes',
            response: 'response',
            // The optional Wald test is omitted: squaring H exceeds its float32 statistic range.
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              fitted: 'fitted',
              chow: 'chow',
              status: 'status'
            }
          })
        ],
        outputs: names
      })
    );
  }
  for (const result of outputs) {
    expect(Array.from(result['status'])).toEqual([0]);
    expectRegressionRowsClose(result['coefficients'], [largeOrigin, 1, 0.5], 1e-6);
    expectRegressionRowsClose(
      result['fitted'],
      [largeOrigin, largeOrigin, largeOrigin, largeOrigin, 0.5, 1.5, 0.5, 1.5],
      1e-6
    );
    expect(result['chow'][0]).toBeNaN();
    expect(Array.from(result['chow'].subarray(1))).toEqual([1, 4]);
  }
  expectRegressionRowsClose(outputs[0]['coefficients'], outputs[1]['coefficients'], 1e-6);
  expectRegressionRowsClose(outputs[0]['fitted'], outputs[1]['fitted'], 1e-6);
  const expectedCovariance = [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.05];
  expectedCovariance.forEach((value, index) => {
    expect(Math.abs(outputs[0]['covariance'][index] - value)).toBeLessThan(1e-7);
    expect(
      Math.abs(outputs[0]['covariance'][index] - outputs[1]['covariance'][index])
    ).toBeLessThan(1e-7);
  });
});

for (const scene of [
  'varying-intercept',
  'common-intercept',
  'no-restrictions',
  'excluded',
  'empty-regime'
] as const) {
  it(`restricted operations agree: ${scene}; relative tolerance 5e-3`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture =
      fixtures[
        scene === 'common-intercept'
          ? 5
          : scene === 'no-restrictions'
            ? 0
            : scene === 'empty-regime'
              ? 7
              : 1
      ];
    const outputs: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
    const names = ['coefficients', 'covariance', 'fitted', 'chow', 'stability', 'status'];
    for (const Operation of [
      GPURestrictedRegimeRegressionOperation,
      CPURestrictedRegimeRegressionOperation
    ]) {
      outputs.push(
        await runOperationGraph(device, {
          inputs: {
            predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
            response: {format: 'float32', values: Float32Array.from(fixture.response)},
            regimes: {
              format: 'uint32',
              values:
                scene === 'empty-regime' ? new Uint32Array(60) : Uint32Array.from(fixture.regimes)
            },
            mask: {
              format: 'uint32',
              values: scene === 'excluded' ? new Uint32Array(60) : Uint32Array.from(fixture.mask)
            }
          },
          operations: [
            new Operation({
              predictorCount: 2,
              regimeCount: 3,
              common: fixture.common,
              rowCount: 60,
              predictors: 'predictors',
              response: 'response',
              regimes: 'regimes',
              mask: 'mask',
              output: {
                coefficients: 'coefficients',
                covariance: 'covariance',
                fitted: 'fitted',
                chow: 'chow',
                stability: 'stability',
                status: 'status'
              }
            })
          ],
          outputs: names
        })
      );
    }
    expect(Array.from(outputs[0]['status'])).toEqual(Array.from(outputs[1]['status']));
    for (const name of ['coefficients', 'fitted', 'chow', 'stability'])
      expectRegressionRowsClose(outputs[0][name], outputs[1][name], 5e-3);
    for (const slot of [1, 2]) {
      const expected = outputs[1]['chow'][slot];
      if (!Number.isNaN(expected)) expect(outputs[0]['chow'][slot]).toBe(expected);
    }
    for (let test = 0; test < 4; test++) {
      expect(outputs[0]['stability'][test * 3 + 1]).toBe(outputs[1]['stability'][test * 3 + 1]);
    }
    const count = outputs[1]['coefficients'].length;
    outputs[1]['covariance'].forEach((value, index) => {
      if (Number.isNaN(value)) {
        expect(outputs[0]['covariance'][index]).toBeNaN();
        return;
      }
      const scale = Math.sqrt(
        outputs[1]['covariance'][Math.floor(index / count) * (count + 1)] *
          outputs[1]['covariance'][(index % count) * (count + 1)]
      );
      expect(Math.abs(outputs[0]['covariance'][index] - value)).toBeLessThanOrEqual(5e-4 * scale);
    });
  });
}
