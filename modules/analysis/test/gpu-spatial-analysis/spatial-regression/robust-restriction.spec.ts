// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURestrictedRegimeRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression-operation';
import {CPURestrictedRegimeRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation';
import {GPUSpatialRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression';
import {getRegimeMomentNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/regime-moments';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {ROBUST_RESTRICTION_FIXTURE as fixture} from './robust-restriction-fixture';

function expectRows(
  actual: ArrayLike<number>,
  expected: readonly (number | null)[],
  tolerance = 3e-4,
  absoluteTolerance = 0
) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, slot) => {
    if (value === null) expect(actual[slot], `slot ${slot}`).toBeNaN();
    else {
      expect(Number.isFinite(actual[slot]), `slot ${slot}`).toBe(true);
      // Float32 small normal solves and fixed-order reductions; the fixture is well conditioned.
      expect(
        Math.abs(actual[slot] - value),
        `slot ${slot}: ${actual[slot]} vs ${value}`
      ).toBeLessThanOrEqual(absoluteTolerance + tolerance * Math.max(1, Math.abs(value)));
    }
  });
}

for (const method of ['white', 'hc1', 'hac'] as const)
  for (const shift of [0, 1e6]) {
    it(`matches independent dense ${method} inference and every legacy port with predictor shift ${shift}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const Operation of [
        GPURestrictedRegimeRegressionOperation,
        CPURestrictedRegimeRegressionOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs: {
            predictors: {
              format: 'float32',
              values: Float32Array.from(
                fixture.predictors,
                (value, slot) => value + (slot % 2 === 0 ? shift : 0)
              )
            },
            response: {format: 'float32', values: Float32Array.from(fixture.response)},
            regimes: {format: 'uint32', values: Uint32Array.from(fixture.regimes)},
            mask: {format: 'uint32', values: Uint32Array.from(fixture.mask)},
            kernelOffsets: {format: 'uint32', values: Uint32Array.from(fixture.offsets)},
            kernelNeighbors: {format: 'uint32', values: Uint32Array.from(fixture.neighbors)},
            kernelWeights: {format: 'float32', values: Float32Array.from(fixture.weights)}
          },
          operations: [
            new Operation({
              predictorCount: 2,
              regimeCount: 3,
              common: [false, true, false],
              rowCount: 33,
              predictors: 'predictors',
              response: 'response',
              regimes: 'regimes',
              mask: 'mask',
              robust:
                method === 'hac'
                  ? {
                      type: 'hac',
                      kernelWeights: {
                        offsets: 'kernelOffsets',
                        neighbors: 'kernelNeighbors',
                        weights: 'kernelWeights'
                      }
                    }
                  : {type: 'white', smallSample: method === 'hc1'},
              output: {
                coefficients: 'coefficients',
                covariance: 'covariance',
                fitted: 'fitted',
                chow: 'chow',
                stability: 'stability',
                status: 'status',
                robust: {
                  covariance: 'robustCovariance',
                  chow: 'robustChow',
                  stability: 'robustStability',
                  status: 'robustStatus'
                }
              }
            })
          ],
          outputs: [
            'coefficients',
            'covariance',
            'fitted',
            'chow',
            'stability',
            'status',
            'robustCovariance',
            'robustChow',
            'robustStability',
            'robustStatus'
          ]
        });
        expectRows(result['status'], [0]);
        expectRows(result['robustStatus'], [0]);
        // Legacy fitted evaluates raw float32 coefficients: a shifted predictor cancels
        // terms near 500,000, with about 0.03 absolute rounding per term. Wald stays shifted.
        expectRows(result['fitted'], fixture.expected.fitted, 3e-4, shift ? 0.125 : 0);
        for (const port of ['chow', 'stability'] as const)
          expectRows(result[port], fixture.expected[port]);
        expectRows(result['robustChow'], fixture.expected[method].chow);
        expectRows(result['robustStability'], fixture.expected[method].stability);
        const coefficients = fixture.expected.coefficients.map(
          (value, slot) => value - (slot < 3 ? shift * fixture.expected.coefficients[3] : 0)
        );
        expectRows(result['coefficients'], coefficients);
        // Raw covariance equivariance is a closed-form linear transformation, not a second fit.
        const transform = Array.from({length: 7}, (_, row) =>
          Array.from(
            {length: 7},
            (_, column) => Number(row === column) - (row < 3 && column === 3 ? shift : 0)
          )
        );
        for (const [port, covariance] of [
          ['covariance', fixture.expected.covariance],
          ['robustCovariance', fixture.expected[method].covariance]
        ] as const) {
          const expected = transform.flatMap(row =>
            transform.map(column =>
              row.reduce(
                (sum, left, first) =>
                  sum +
                  column.reduce(
                    (innerSum, right, second) =>
                      innerSum + left * right * covariance[first * 7 + second],
                    0
                  ),
                0
              )
            )
          );
          expectRows(result[port], expected);
        }
      }
    });
  }

for (const model of ['commonIntercept', 'pivot'] as const)
  for (const method of ['white', 'hc1', 'hac'] as const) {
    it(`matches independent ${model} ${method} inference on every output`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const Operation of [
        GPURestrictedRegimeRegressionOperation,
        CPURestrictedRegimeRegressionOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs: {
            predictors: {
              format: 'float32',
              values: Float32Array.from(
                fixture.predictors,
                (value, slot) => value + (model === 'pivot' && slot % 2 === 1 ? 1e6 : 0)
              )
            },
            response: {format: 'float32', values: Float32Array.from(fixture.response)},
            regimes: {format: 'uint32', values: Uint32Array.from(fixture.regimes)},
            mask: {format: 'uint32', values: Uint32Array.from(fixture.mask)},
            kernelOffsets: {format: 'uint32', values: Uint32Array.from(fixture.offsets)},
            kernelNeighbors: {format: 'uint32', values: Uint32Array.from(fixture.neighbors)},
            kernelWeights: {format: 'float32', values: Float32Array.from(fixture.weights)}
          },
          operations: [
            new Operation({
              predictorCount: 2,
              regimeCount: 3,
              common: [true, true, false],
              rowCount: 33,
              predictors: 'predictors',
              response: 'response',
              regimes: 'regimes',
              mask: 'mask',
              robust:
                method === 'hac'
                  ? {
                      type: 'hac',
                      kernelWeights: {
                        offsets: 'kernelOffsets',
                        neighbors: 'kernelNeighbors',
                        weights: 'kernelWeights'
                      }
                    }
                  : {type: 'white', smallSample: method === 'hc1'},
              output: {
                coefficients: 'coefficients',
                covariance: 'covariance',
                fitted: 'fitted',
                chow: 'chow',
                stability: 'stability',
                status: 'status',
                robust: {
                  covariance: 'robustCovariance',
                  chow: 'robustChow',
                  stability: 'robustStability',
                  status: 'robustStatus'
                }
              }
            })
          ],
          outputs: [
            'coefficients',
            'covariance',
            'fitted',
            'chow',
            'stability',
            'status',
            'robustCovariance',
            'robustChow',
            'robustStability',
            'robustStatus'
          ]
        });
        for (const port of [
          'coefficients',
          'covariance',
          'fitted',
          'chow',
          'stability',
          'status'
        ] as const)
          expectRows(
            result[port],
            fixture[model][port],
            5e-4,
            model === 'pivot' && port === 'fitted' ? 0.125 : 0
          );
        for (const [port, expected] of [
          ['robustCovariance', fixture[model][method].covariance],
          ['robustChow', fixture[model][method].chow],
          ['robustStability', fixture[model][method].stability],
          ['robustStatus', fixture[model][method].status]
        ] as const)
          expectRows(result[port], expected, 5e-4);
      }
    });
  }

for (const defect of [
  'overflow',
  'candidateOverflow',
  'converged',
  'diagonal',
  'asymmetric',
  'indefinite',
  'masked'
] as const) {
  it(`suppresses robust results with explicit status for ${defect}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const weights = Float32Array.from(fixture.weights, value =>
      defect === 'indefinite' && value !== 1 ? 100 * value : value
    );
    if (defect === 'diagonal') weights[0] = 0.5;
    if (defect === 'asymmetric') weights[1] = 0.125;
    for (const Operation of [
      GPURestrictedRegimeRegressionOperation,
      CPURestrictedRegimeRegressionOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
          response: {format: 'float32', values: Float32Array.from(fixture.response)},
          regimes: {format: 'uint32', values: Uint32Array.from(fixture.regimes)},
          mask: {
            format: 'uint32',
            values: defect === 'masked' ? new Uint32Array(33) : Uint32Array.from(fixture.mask)
          },
          kernelOffsets: {format: 'uint32', values: Uint32Array.from(fixture.offsets)},
          kernelNeighbors: {format: 'uint32', values: Uint32Array.from(fixture.neighbors)},
          kernelWeights: {format: 'float32', values: weights},
          kernelOverflow: {
            format: 'uint32',
            values: Uint32Array.of(defect === 'overflow' || defect === 'candidateOverflow' ? 1 : 0)
          }
        },
        operations: [
          new Operation({
            rowCount: 33,
            predictorCount: 2,
            regimeCount: 3,
            common: [false, true, false],
            predictors: 'predictors',
            response: 'response',
            regimes: 'regimes',
            mask: 'mask',
            robust: {
              type: 'hac',
              kernelWeights: {
                offsets: 'kernelOffsets',
                neighbors: 'kernelNeighbors',
                weights: 'kernelWeights',
                status: {
                  [defect === 'converged'
                    ? 'converged'
                    : defect === 'candidateOverflow'
                      ? 'candidateOverflow'
                      : 'overflow']: 'kernelOverflow'
                }
              }
            },
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              fitted: 'fitted',
              chow: 'chow',
              status: 'status',
              robust: {
                covariance: 'robustCovariance',
                chow: 'robustChow',
                stability: 'robustStability',
                status: 'robustStatus'
              }
            }
          })
        ],
        outputs: ['robustCovariance', 'robustChow', 'robustStability', 'robustStatus']
      });
      expectRows(result['robustStatus'], [
        defect === 'masked' ? 1 : defect === 'indefinite' ? 3 : 2
      ]);
      expect(Array.from(result['robustCovariance']).every(Number.isNaN)).toBe(true);
      expectRows(result['robustChow'], [null, 0, null]);
      expectRows(result['robustStability'], [
        null,
        0,
        null,
        null,
        0,
        null,
        null,
        0,
        null,
        null,
        0,
        null
      ]);
    }
  });
}

for (const shift of [0, 1e6])
  it(`matches independent per-regime OLS and moments with exclusions, empty group and shift ${shift}`, async () => {
    const context = await createRegressionTestContext();
    const {graph, input, output} = context;
    try {
      const props = {
        id: 'oracle-spatial',
        predictors: input(
          'float32',
          fixture.predictors.map((value, slot) => value + (slot % 2 === 0 ? shift : 0))
        ),
        predictorCount: 2,
        response: input('float32', fixture.response),
        regimes: input('uint32', fixture.regimes),
        regimeCount: 4,
        mask: input('uint32', fixture.mask)
      };
      const coefficients = output('float32', 12);
      const fitted = output('float32', 33);
      const status = output('uint32', 4);
      const counts = output('uint32', 4);
      graph.add(
        new GPUSpatialRegimeRegression({
          ...props,
          output: {
            coefficients: coefficients.view,
            fitted: fitted.view,
            status: status.view,
            counts: counts.view
          }
        })
      );
      const moments = getRegimeMomentNodes(graph, {
        ...props,
        id: 'oracle-moments',
        operation: 'OracleMoments'
      });
      for (const node of moments.nodes) graph.add(node);
      // Copy transient moments to caller-owned readback storage within the same graph.
      const momentOutput = output('float32', 48);
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'oracle-moment-copy',
          operation: 'OracleMoments',
          invocationCount: 48,
          bindings: [
            {name: 'source', view: moments.views.moments, type: 'f32', access: 'read'},
            {name: 'copiedMoments', view: momentOutput.view, type: 'f32', access: 'read_write'}
          ],
          body: 'copiedMoments[copiedMomentsOffset + index] = source[sourceOffset + index];'
        })
      );
      const offsetsOutput = output('uint32', 5);
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'oracle-offset-copy',
          operation: 'OracleMoments',
          invocationCount: 5,
          bindings: [
            {
              name: 'sourceOffsets',
              view: moments.views.segmentOffsets,
              type: 'u32',
              access: 'read'
            },
            {name: 'copiedOffsets', view: offsetsOutput.view, type: 'u32', access: 'read_write'}
          ],
          body: 'copiedOffsets[copiedOffsetsOffset + index] = sourceOffsets[sourceOffsetsOffset + index];'
        })
      );
      context.run();
      const expectedCoefficients = fixture.spatial.coefficients.map((value, slot) =>
        value === null
          ? null
          : value - (slot % 3 === 0 ? shift * fixture.spatial.coefficients[slot + 1]! : 0)
      );
      expectRows(await readFloat32(coefficients.buffer, 12), expectedCoefficients);
      expectRows(await readFloat32(fitted.buffer, 33), fixture.spatial.fitted);
      expectRows(await readUint32(status.buffer, 4), fixture.spatial.status);
      expectRows(await readUint32(counts.buffer, 4), fixture.spatial.counts);
      expectRows(await readUint32(offsetsOutput.buffer, 5), [0, 11, 20, 30, 30]);
      // Exact change of origin applied to oracle literals: C_ab -> C_ab + da*S_b + db*S_a + n*da*db.
      const expectedMoments = fixture.spatial.moments.slice();
      for (let regime = 0; regime < 4; regime++) {
        const count = fixture.spatial.counts[regime];
        if (!count) continue;
        const base = regime * 12;
        const old = fixture.spatial.moments.slice(base, base + 12);
        const newMean = Math.fround(old[0] + old[9] / count + shift);
        const offsets = [old[0] + shift - newMean, 0, 0];
        expectedMoments[base] = newMean;
        expectedMoments[base + 9] += count * offsets[0];
        let moment = 3;
        for (let first = 0; first < 3; first++)
          for (let second = first; second < 3; second++) {
            expectedMoments[base + moment] +=
              offsets[first] * old[9 + second] +
              offsets[second] * old[9 + first] +
              count * offsets[first] * offsets[second];
            moment++;
          }
      }
      expectRows(await readFloat32(momentOutput.buffer, 48), expectedMoments, 2e-5);
    } finally {
      context.destroy();
    }
  });
