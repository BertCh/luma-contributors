// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation';
import {GPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './seemingly-unrelated-regression-operation-fixtures.json';
import spatialFixture from './seemingly-unrelated-regression-spatial-operation-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const scene of ['self-links', 'valid-spatial'] as const) {
  it(`preserves spatial SUR values and statuses for ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = scene === 'self-links' ? 6 : spatialFixture.rowCount;
    const columns = scene === 'self-links' ? 1 : 2;
    const names = [
      'coefficients',
      'covariance',
      'residualCovariance',
      'residuals',
      'status',
      'parameters',
      'spatialStatus'
    ];
    const outputs: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
    for (const Operation of [
      GPUSeeminglyUnrelatedRegressionOperation,
      CPUSeeminglyUnrelatedRegressionOperation
    ]) {
      outputs.push(
        await runOperationGraph(device, {
          inputs: {
            predictors: {
              format: 'float32',
              values:
                scene === 'self-links'
                  ? Float32Array.of(-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3)
                  : Float32Array.from(spatialFixture.predictors)
            },
            responses: {
              format: 'float32',
              values:
                scene === 'self-links'
                  ? Float32Array.of(0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 2, 0)
                  : Float32Array.from(spatialFixture.responses)
            },
            offsets: {
              format: 'uint32',
              values:
                scene === 'self-links'
                  ? Uint32Array.of(0, 1, 2, 3, 4, 5, 6)
                  : Uint32Array.from(spatialFixture.weights.offsets)
            },
            neighbors: {
              format: 'uint32',
              values:
                scene === 'self-links'
                  ? Uint32Array.of(0, 1, 2, 3, 4, 5)
                  : Uint32Array.from(spatialFixture.weights.neighbors)
            },
            weights: {
              format: 'float32',
              values:
                scene === 'self-links'
                  ? new Float32Array(6).fill(1)
                  : Float32Array.from(spatialFixture.weights.weights)
            }
          },
          operations: [
            new Operation({
              rowCount: rows,
              predictorCount: columns,
              equationCount: 2,
              predictors: 'predictors',
              responses: 'responses',
              spatial: {
                weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
                output: {parameters: 'parameters', status: 'spatialStatus'}
              },
              output: {
                coefficients: 'coefficients',
                covariance: 'covariance',
                residualCovariance: 'residualCovariance',
                residuals: 'residuals',
                status: 'status'
              }
            })
          ],
          outputs: names
        })
      );
    }
    expect(Array.from(outputs[0]['status'])).toEqual(Array.from(outputs[1]['status']));
    expect(Array.from(outputs[0]['spatialStatus'])).toEqual(
      Array.from(outputs[1]['spatialStatus'])
    );
    for (const name of ['coefficients', 'residualCovariance', 'residuals', 'parameters'])
      expectRegressionRowsClose(outputs[0][name], outputs[1][name], 5e-3);
    const size = outputs[1]['coefficients'].length;
    outputs[1]['covariance'].forEach((value, index) => {
      if (Number.isNaN(value)) {
        expect(outputs[0]['covariance'][index]).toBeNaN();
        return;
      }
      const scale = Math.sqrt(
        outputs[1]['covariance'][Math.floor(index / size) * (size + 1)] *
          outputs[1]['covariance'][(index % size) * (size + 1)]
      );
      expect(Math.abs(outputs[0]['covariance'][index] - value)).toBeLessThanOrEqual(5e-3 * scale);
    });
    if (scene === 'self-links') {
      for (const result of outputs) {
        expect(Array.from(result['spatialStatus'])).toEqual([4, 4]);
        expect(result['parameters'].every(Number.isNaN)).toBe(true);
        expect(Array.from(result['status'])).toEqual([1]);
        for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'])
          expect(result[name].every(Number.isNaN)).toBe(true);
      }
    } else {
      for (const result of outputs) {
        expect(Array.from(result['spatialStatus'])).toEqual([0, 0]);
        expect(Array.from(result['status'])).toEqual([0]);
      }
    }
  });
}

for (const scene of ['two', 'three', 'supplied', 'excluded', 'singular-covariance'] as const) {
  it(`SUR operations agree: ${scene} (relative tolerance 5e-3)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures[scene === 'three' ? 1 : scene === 'supplied' ? 2 : 0];
    const supplied = scene === 'supplied' || scene === 'singular-covariance';
    const names = ['coefficients', 'covariance', 'residualCovariance', 'residuals', 'status'];
    const outputs: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
    for (const Operation of [
      GPUSeeminglyUnrelatedRegressionOperation,
      CPUSeeminglyUnrelatedRegressionOperation
    ]) {
      outputs.push(
        await runOperationGraph(device, {
          inputs: {
            predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
            responses: {format: 'float32', values: Float32Array.from(fixture.responses)},
            mask: {
              format: 'uint32',
              values: scene === 'excluded' ? new Uint32Array(40) : Uint32Array.from(fixture.mask)
            },
            ...(supplied
              ? {
                  initial: {
                    format: 'float32' as const,
                    values:
                      scene === 'singular-covariance'
                        ? new Float32Array(80).fill(2)
                        : Float32Array.from(fixture.initialResiduals!)
                  }
                }
              : {})
          },
          operations: [
            new Operation({
              equationCount: fixture.equationCount,
              predictorCount: 2,
              rowCount: 40,
              predictors: 'predictors',
              responses: 'responses',
              mask: 'mask',
              initialResiduals: supplied ? 'initial' : undefined,
              output: {
                coefficients: 'coefficients',
                covariance: 'covariance',
                residualCovariance: 'residualCovariance',
                residuals: 'residuals',
                status: 'status'
              }
            })
          ],
          outputs: names
        })
      );
    }
    expect(Array.from(outputs[0]['status'])).toEqual(Array.from(outputs[1]['status']));
    for (const name of ['coefficients', 'residualCovariance', 'residuals'])
      expectRegressionRowsClose(outputs[0][name], outputs[1][name], 5e-3);
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
      expect(Math.abs(outputs[0]['covariance'][index] - value)).toBeLessThanOrEqual(5e-3 * scale);
    });
  });
}
