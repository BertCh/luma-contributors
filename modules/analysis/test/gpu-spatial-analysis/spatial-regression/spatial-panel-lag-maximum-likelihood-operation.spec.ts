// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialPanelLagMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';
import {GPUSpatialPanelLagMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import fixtures from './spatial-panel-lag-maximum-likelihood-operation-fixtures.json';
import edges from './spatial-panel-lag-maximum-likelihood-operation-edge-fixtures.json';
import {panelPoleScenes} from './panel-lag-poles.test-utils';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';

for (const scene of panelPoleScenes)
  for (const route of [
    'unit',
    'random',
    ...(['weightedPole', 'weightedRationalPole', 'rationalSlopePole'].includes(scene.name)
      ? (['tiled'] as const)
      : [])
  ] as const)
    for (const affine of [false, true])
      it(`all panel lag forms reject the Fraction-certified pole ${scene.name}, ${route}, affine=${affine}`, async () => {
        const effects = route === 'random' ? 'random' : 'unit';
        const predictors = Float32Array.from(scene.predictors, value => value + (affine ? 8 : 0));
        const response = Float32Array.from(scene.response, (value, row) =>
          affine ? 2 * value + 16 * (row % 4) : value
        );
        const context = await createRegressionTestContext();
        try {
          const coefficients = context.output('float32', scene.predictorCount + 3),
            covariance = context.output('float32', (scene.predictorCount + 2) ** 2);
          const summary = context.output('float32', 6),
            fitted = context.output('float32', 8),
            residuals = context.output('float32', 8),
            status = context.output('uint32', 1);
          const hausman =
            effects === 'random'
              ? {tests: context.output('float32', 3), status: context.output('uint32', 1)}
              : undefined;
          context.graph.add(
            new GPUSpatialPanelLagMaximumLikelihood({
              periodCount: 2,
              predictorCount: scene.predictorCount,
              effects,
              eigenvalues:
                route === 'tiled'
                  ? context.input(
                      'float32',
                      scene.name.startsWith('weighted') ? [-1, -0.25, 0.25, 1] : [-1, 0, 0, 1]
                    )
                  : undefined,
              predictors: context.input('float32', predictors),
              response: context.input('float32', response),
              weights: {
                offsets: context.input('uint32', scene.weights.offsets),
                neighbors: context.input('uint32', scene.weights.neighbors),
                weights: context.input('float32', scene.weights.weights)
              },
              output: {
                coefficients: coefficients.view,
                covariance: covariance.view,
                summary: summary.view,
                fitted: fitted.view,
                residuals: residuals.view,
                status: status.view
              },
              hausman: hausman
                ? {tests: hausman.tests.view, status: hausman.status.view}
                : undefined
            })
          );
          context.run();
          expect(Array.from(await readUint32(status.buffer, 1))).toEqual([2]);
          for (const port of [coefficients, covariance, summary, fitted, residuals])
            expect((await readFloat32(port.buffer, port.view.length)).every(Number.isNaN)).toBe(
              true
            );
          if (hausman) {
            expect(Array.from(await readUint32(hausman.status.buffer, 1))).toEqual([1]);
            expect((await readFloat32(hausman.tests.buffer, 3)).every(Number.isNaN)).toBe(true);
          }
          for (const Operation of [
            GPUSpatialPanelLagMaximumLikelihoodOperation,
            CPUSpatialPanelLagMaximumLikelihoodOperation
          ]) {
            const result = await runOperationGraph(context.device, {
              inputs: {
                predictors: {format: 'float32', values: predictors},
                response: {format: 'float32', values: response},
                offsets: {format: 'uint32', values: Uint32Array.from(scene.weights.offsets)},
                neighbors: {format: 'uint32', values: Uint32Array.from(scene.weights.neighbors)},
                weights: {format: 'float32', values: Float32Array.from(scene.weights.weights)},
                ...(route === 'tiled'
                  ? {
                      eigenvalues: {
                        format: 'float32' as const,
                        values: Float32Array.from(
                          scene.name.startsWith('weighted') ? [-1, -0.25, 0.25, 1] : [-1, 0, 0, 1]
                        )
                      }
                    }
                  : {})
              },
              operations: [
                new Operation({
                  entityCount: 4,
                  periodCount: 2,
                  predictorCount: scene.predictorCount,
                  effects,
                  eigenvalues: route === 'tiled' ? 'eigenvalues' : undefined,
                  predictors: 'predictors',
                  response: 'response',
                  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
                  output: {
                    coefficients: 'coefficients',
                    covariance: 'covariance',
                    summary: 'summary',
                    status: 'status',
                    fitted: 'fitted',
                    residuals: 'residuals'
                  },
                  hausman:
                    effects === 'random'
                      ? {tests: 'hausmanTests', status: 'hausmanStatus'}
                      : undefined
                })
              ],
              outputs: [
                'coefficients',
                'covariance',
                'summary',
                'status',
                'fitted',
                'residuals',
                ...(effects === 'random' ? ['hausmanTests', 'hausmanStatus'] : [])
              ]
            });
            expect(Array.from(result['status'])).toEqual([2]);
            for (const port of [
              'coefficients',
              'covariance',
              'summary',
              'fitted',
              'residuals',
              ...(effects === 'random' ? ['hausmanTests'] : [])
            ])
              expect(Array.from(result[port]).every(Number.isNaN)).toBe(true);
            if (effects === 'random') expect(Array.from(result['hausmanStatus'])).toEqual([1]);
          }
        } finally {
          context.destroy();
        }
      });

for (const scene of [
  'random',
  'unit',
  'phiOne',
  'invalidWeights',
  'overflow',
  'singular',
  'optionalWeights'
] as const) {
  it(`panel lag ML operations compare every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures[scene === 'phiOne' ? 2 : 0];
    const unit = scene === 'unit';
    const outputNames = [
      'coefficients',
      'covariance',
      'summary',
      'status',
      'fitted',
      'residuals',
      ...(!unit ? ['hausmanTests', 'hausmanStatus'] : [])
    ];
    const results = [];
    for (const Operation of [
      GPUSpatialPanelLagMaximumLikelihoodOperation,
      CPUSpatialPanelLagMaximumLikelihoodOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            predictors: {
              format: 'float32',
              values:
                scene === 'singular'
                  ? new Float32Array(fixture.predictors.length)
                  : Float32Array.from(fixture.predictors)
            },
            response: {format: 'float32', values: Float32Array.from(fixture.response)},
            offsets: {format: 'uint32', values: Uint32Array.from(fixture.weights.offsets)},
            neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
            weights: {
              format: 'float32',
              values:
                scene === 'invalidWeights'
                  ? new Float32Array(fixture.weights.weights.length)
                  : Float32Array.from(fixture.weights.weights)
            },
            overflow: {format: 'uint32', values: Uint32Array.of(scene === 'overflow' ? 1 : 0)},
            ...(scene === 'optionalWeights'
              ? {
                  distances: {
                    format: 'float32' as const,
                    values: new Float32Array(fixture.weights.neighbors.length)
                  },
                  requiredCount: {
                    format: 'uint32' as const,
                    values: Uint32Array.of(fixture.weights.neighbors.length)
                  },
                  candidateOverflow: {format: 'uint32' as const, values: Uint32Array.of(0)},
                  approximation: {format: 'uint32' as const, values: Uint32Array.of(1)}
                }
              : {})
          },
          // Every call owns a fresh operation: runOperationGraph destroys it.
          operations: [
            new Operation({
              entityCount: fixture.entityCount,
              periodCount: fixture.periodCount,
              predictorCount: fixture.predictorCount,
              effects: unit ? 'unit' : 'random',
              tolerance: 1e-5,
              predictors: 'predictors',
              response: 'response',
              weights: {
                offsets: 'offsets',
                neighbors: 'neighbors',
                weights: 'weights',
                status: {overflow: 'overflow'},
                ...(scene === 'optionalWeights'
                  ? {
                      distances: 'distances',
                      status: {
                        overflow: 'overflow',
                        requiredCount: 'requiredCount',
                        candidateOverflow: 'candidateOverflow',
                        approximation: 'approximation'
                      },
                      information: {
                        rowCount: fixture.entityCount,
                        transform: 'row' as const,
                        symmetric: false
                      },
                      spatialContext: {
                        coordinateSpace: 'planar' as const,
                        metric: 'native' as const
                      }
                    }
                  : {})
              },
              output: {
                coefficients: 'coefficients',
                covariance: 'covariance',
                summary: 'summary',
                status: 'status',
                fitted: 'fitted',
                residuals: 'residuals'
              },
              hausman: unit ? undefined : {tests: 'hausmanTests', status: 'hausmanStatus'}
            })
          ],
          outputs: outputNames
        })
      );
    }
    const [gpu, cpu] = results;
    const failure = ['invalidWeights', 'overflow', 'singular'].includes(scene);
    expect(cpu['status'][0]).toBe(scene === 'singular' ? 2 : failure ? 1 : 0);
    expectRegressionRowsClose(gpu['status'], cpu['status'], 0);
    if (failure) {
      for (const name of outputNames) expectRegressionRowsClose(gpu[name], cpu[name], 0);
      return;
    }
    // Nontrivial multiple predictors, asymmetric unequal weights and six periods. Relative
    // parameter/likelihood tolerance 0.5%; covariance error uses diagonal geometric means.
    expectRegressionRowsClose(gpu['coefficients'], cpu['coefficients'], 5e-3);
    expectRegressionRowsClose(gpu['summary'].subarray(0, 2), cpu['summary'].subarray(0, 2), 5e-3);
    const width = fixture.predictorCount + 2;
    for (let row = 0; row < width; row++)
      for (let column = 0; column < width; column++) {
        const expected = cpu['covariance'][row * width + column];
        if (Number.isNaN(expected)) expect(gpu['covariance'][row * width + column]).toBeNaN();
        else
          expect(Math.abs(gpu['covariance'][row * width + column] - expected)).toBeLessThanOrEqual(
            5e-3 *
              Math.sqrt(
                cpu['covariance'][row * width + row] * cpu['covariance'][column * width + column]
              )
          );
      }
    // Sweep counts describe separate float32 and float64 convergence histories. There is
    // no bound on their difference. Validate each count's own budget/meaning instead.
    expect(Number.isInteger(gpu['summary'][2])).toBe(true);
    expect(gpu['summary'][2]).toBeGreaterThanOrEqual(1);
    expect(gpu['summary'][2]).toBeLessThanOrEqual(unit ? 1 : 64);
    expect(Number.isInteger(cpu['summary'][2])).toBe(true);
    expect(cpu['summary'][2]).toBeGreaterThanOrEqual(1);
    if (unit) expect(cpu['summary'][2]).toBe(1);
    // Both successful fits meet their declared phi tolerance; 32 GPU bracket steps leave
    // width <=5e-7 while the uncapped CPU search reaches binary64 precision.
    expect(Math.abs(gpu['summary'][3] - cpu['summary'][3])).toBeLessThanOrEqual(1e-5);
    expect(Math.abs(gpu['summary'][4] - cpu['summary'][4])).toBeLessThanOrEqual(5e-7);
    expect(gpu['summary'][5] === cpu['summary'][5]).toBe(true);
    const responseScale = Math.sqrt(
      fixture.response.reduce((sum, value) => sum + value * value, 0) / fixture.response.length
    );
    for (const port of ['fitted', 'residuals']) {
      expect(gpu[port].length).toBe(cpu[port].length);
      for (let row = 0; row < fixture.response.length; row++)
        expect(Math.abs(gpu[port][row] - cpu[port][row])).toBeLessThanOrEqual(
          2e-4 * Math.max(responseScale, Math.abs(cpu[port][row]))
        );
    }
    if (!unit) {
      expectRegressionRowsClose(gpu['hausmanStatus'], cpu['hausmanStatus'], 0);
      expectRegressionRowsClose(gpu['hausmanTests'], cpu['hausmanTests'], 5e-3);
      expect(cpu['hausmanStatus'][0]).toBe(scene === 'phiOne' ? 2 : 0);
      if (scene === 'random') {
        // Moderate p-value pins the probability computation, not just a tail rounded to zero.
        expect(cpu['hausmanTests'][2]).toBeGreaterThan(0.3);
        expect(cpu['hausmanTests'][2]).toBeLessThan(0.5);
      }
    }
  });
}

for (const scene of edges.reviewCases) {
  it(`panel operations honor independent review edge expectations: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const outputNames = ['coefficients', 'covariance', 'summary', 'status', 'fitted', 'residuals'];
    const results = [];
    for (const Operation of [
      GPUSpatialPanelLagMaximumLikelihoodOperation,
      CPUSpatialPanelLagMaximumLikelihoodOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            offsets: {format: 'uint32', values: Uint32Array.from(scene.weights.offsets)},
            neighbors: {format: 'uint32', values: Uint32Array.from(scene.weights.neighbors)},
            weights: {format: 'float32', values: Float32Array.from(scene.weights.weights)},
            predictors: {format: 'float32', values: Float32Array.from(scene.predictors)},
            response: {format: 'float32', values: Float32Array.from(scene.response)}
          },
          operations: [
            new Operation({
              entityCount: scene.entityCount,
              periodCount: scene.periodCount,
              predictorCount: scene.predictorCount,
              predictors: 'predictors',
              response: 'response',
              effects: 'unit',
              weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
              output: {
                coefficients: 'coefficients',
                covariance: 'covariance',
                summary: 'summary',
                status: 'status',
                fitted: 'fitted',
                residuals: 'residuals'
              }
            })
          ],
          outputs: outputNames
        })
      );
    }
    const [gpu, cpu] = results;
    const pole = scene.name === 'responseCopy' || scene.name === 'originalPole';
    const invalidWeights = scene.name === 'normalizationBoundary';
    expect(cpu['status'][0]).toBe(invalidWeights ? 1 : 2);
    // Include every observed GPU output when an independent edge expectation fails.
    const observed = JSON.stringify(
      Object.fromEntries(
        outputNames.map(port => [
          port,
          Array.from(gpu[port], value => (Number.isFinite(value) ? value : String(value)))
        ])
      )
    );
    expect(gpu['status'][0], `GPU review edge ${scene.name}: ${observed}`).toBe(
      invalidWeights ? 1 : 2
    );
    for (const port of ['covariance', 'fitted', 'residuals'])
      expect(cpu[port].every(Number.isNaN)).toBe(true);
    if (invalidWeights) {
      // Both reject the same serial-f32 row sum: every output, including NaNs, agrees exactly.
      for (const port of outputNames) expectRegressionRowsClose(gpu[port], cpu[port], 0);
    } else if (pole) {
      // Fraction oracle: responseCopy has beta=[1,0], rho=0; originalPole beta=1,
      // rho=.5. Both have exact zero within variance and no finite Gaussian maximum.
      for (const port of outputNames.filter(port => port !== 'status')) {
        expect(Array.from(gpu[port]).every(Number.isNaN)).toBe(true);
        expect(Array.from(cpu[port]).every(Number.isNaN)).toBe(true);
      }
    } else {
      // scale²/2 is outside float32. The CPU preserves a finite float64 likelihood while
      // the GPU can overflow/underflow earlier; those failure diagnostics need not agree.
      const expectedVariance = scene.name === 'varianceOverflow' ? Infinity : 0;
      expect(cpu['summary'][0] === expectedVariance).toBe(true);
      if (scene.name === 'varianceUnderflow') {
        // Test correction: the GPU's squared moments underflow before its profile search.
        // The early status-2 path withholds all diagnostics. CPU float64 retains a finite
        // likelihood before its variance rounds to zero at float32 publication.
        for (const port of outputNames.filter(port => port !== 'status'))
          expect(Array.from(gpu[port]).every(Number.isNaN), `${port}: ${observed}`).toBe(true);
      } else expect(gpu['summary'][0] === expectedVariance).toBe(true);
      for (const port of ['covariance', 'fitted', 'residuals'])
        expectRegressionRowsClose(gpu[port], cpu[port], 0);
    }
  });
}

it('both operations reject the review four-slot CSR with a one-row distances view', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = edges.reviewCases[0];
  for (const Operation of [
    GPUSpatialPanelLagMaximumLikelihoodOperation,
    CPUSpatialPanelLagMaximumLikelihoodOperation
  ]) {
    await expect(
      runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.from(scene.weights.offsets)},
          neighbors: {format: 'uint32', values: Uint32Array.from(scene.weights.neighbors)},
          weights: {format: 'float32', values: Float32Array.from(scene.weights.weights)},
          distances: {format: 'float32', values: Float32Array.of(1)},
          predictors: {format: 'float32', values: Float32Array.from(scene.predictors)},
          response: {format: 'float32', values: Float32Array.from(scene.response)}
        },
        operations: [
          new Operation({
            entityCount: 4,
            periodCount: 2,
            predictorCount: 1,
            effects: 'unit',
            predictors: 'predictors',
            response: 'response',
            weights: {
              offsets: 'offsets',
              neighbors: 'neighbors',
              weights: 'weights',
              distances: 'distances'
            },
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              summary: 'summary',
              status: 'status',
              fitted: 'fitted',
              residuals: 'residuals'
            }
          })
        ],
        outputs: ['status']
      })
    ).rejects.toThrow(/weights.distances length/);
  }
});
