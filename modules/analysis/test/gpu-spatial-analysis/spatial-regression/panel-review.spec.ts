// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  PANEL_REGRESSION_AUDIT_EXPECTED,
  PANEL_DIAGNOSTIC_AUDIT_EXPECTED
} from './panel-regression-audit.test-utils';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {getPanelLagPoleNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-lag-pole-nodes';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {panelPoleScenes} from './panel-lag-poles.test-utils';
import {
  createPanelAuditInput,
  expectPanelAuditRows,
  PANEL_AUDIT_EXPECTED
} from './panel-audit.test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {CPUSpatialPanelLagMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';
import {GPUSpatialPanelLagMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood-operation';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {
  createOffsetPanelLagInput,
  expectOffsetPanelLag,
  createConstantPanelLagInput,
  createBoundaryPanelLagInput,
  expectBoundaryPanelLag,
  createExtremeOffsetPanelLagInput,
  EXTREME_OFFSET_PANEL_LAG_EXPECTED,
  createExtremeFirstPanelLagInput
} from './panel-review.test-utils';

for (const budget of [1, 32, 64])
  it(`R6 operation boundary optimum respects search budget ${budget}`, async () => {
    const context = await createRegressionTestContext(),
      input = createBoundaryPanelLagInput();
    try {
      for (const Operation of [
        GPUSpatialPanelLagMaximumLikelihoodOperation,
        CPUSpatialPanelLagMaximumLikelihoodOperation
      ]) {
        const result = await runOperationGraph(context.device, {
          inputs: {
            predictors: {format: 'float32', values: input.predictors},
            response: {format: 'float32', values: input.response},
            offsets: {format: 'uint32', values: input.weights.offsets},
            neighbors: {format: 'uint32', values: input.weights.neighbors},
            weights: {format: 'float32', values: input.weights.weights}
          },
          operations: [
            new Operation({
              entityCount: 3,
              periodCount: 4,
              predictorCount: 1,
              effects: 'unit',
              maximumSearchIterations: budget,
              predictors: 'predictors',
              response: 'response',
              weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
              output: {
                coefficients: 'coefficients',
                covariance: 'covariance',
                summary: 'summary',
                fitted: 'fitted',
                residuals: 'residuals',
                status: 'status'
              }
            })
          ],
          outputs: ['coefficients', 'covariance', 'summary', 'fitted', 'residuals', 'status']
        });
        expectBoundaryPanelLag(
          {
            coefficients: result['coefficients'],
            covariance: result['covariance'],
            summary: result['summary'],
            fitted: result['fitted'],
            residuals: result['residuals'],
            status: result['status']
          },
          budget
        );
      }
    } finally {
      context.destroy();
    }
  });

it('review 2 R5 native pack preserves the four signed two-thirds within values', async () => {
  const context = await createRegressionTestContext(),
    input = createExtremeFirstPanelLagInput();
  try {
    recordWGSLKernelNodes(context.graph);
    const contributor = new GPUSpatialPanelLagMaximumLikelihood({
      effects: 'unit',
      periodCount: 3,
      predictorCount: 1,
      predictors: context.input('float32', input.predictors),
      response: context.input('float32', input.response),
      weights: {
        offsets: context.input('uint32', input.weights.offsets),
        neighbors: context.input('uint32', input.weights.neighbors),
        weights: context.input('float32', input.weights.weights)
      },
      output: {
        coefficients: context.output('float32', 4).view,
        covariance: context.output('float32', 9).view,
        summary: context.output('float32', 6).view,
        status: context.output('uint32', 1).view,
        fitted: context.output('float32', 12).view,
        residuals: context.output('float32', 12).view
      }
    });
    const nodes = contributor
      .getCommandNodes(context.graph)
      .filter(
        node =>
          node.id.includes('-within-pack-') ||
          ['-pooled-means', '-dense-weights', '-pack'].some(suffix => node.id.endsWith(suffix))
      );
    const record = getWGSLKernelNodeRecord(
      context.graph,
      nodes.find(node => node.id.endsWith('-pack'))!.id
    )!;
    const output = context.output('float32', 4);
    context.graph.add(nodes);
    context.graph.add(
      createWGSLKernelNode(context.graph, {
        id: 'r2-r5-read-within',
        operation: 'PanelReview',
        invocationCount: 4,
        bindings: [
          record.bindings.find(binding => binding.name === 'data')!,
          {name: 'output', view: output.view, type: 'f32', access: 'read_write'}
        ],
        body: 'output[outputOffset+index]=data[dataOffset+(index+8u)*8u+1u];'
      })
    );
    context.run();
    const values = await readFloat32(output.buffer, 4);
    for (let entity = 0; entity < 4; entity++) {
      expect(Number.isFinite(values[entity])).toBe(true);
      expect(Math.abs(values[entity] - (entity % 2 ? -2 / 3 : 2 / 3))).toBeLessThan(2 ** -24);
    }
  } finally {
    context.destroy();
  }
});

it('R5 native pack has twelve zero within values and the reference unit fit refuses rank zero', async () => {
  const context = await createRegressionTestContext(),
    input = createConstantPanelLagInput();
  try {
    recordWGSLKernelNodes(context.graph);
    const output = {
      coefficients: context.output('float32', 4),
      covariance: context.output('float32', 9),
      summary: context.output('float32', 6),
      fitted: context.output('float32', 12),
      residuals: context.output('float32', 12)
    };
    const status = context.output('uint32', 1),
      within = context.output('float32', 12);
    const contributor = new GPUSpatialPanelLagMaximumLikelihood({
      effects: 'unit',
      periodCount: 3,
      predictorCount: 1,
      predictors: context.input('float32', input.predictors),
      response: context.input('float32', input.response),
      weights: {
        offsets: context.input('uint32', input.weights.offsets),
        neighbors: context.input('uint32', input.weights.neighbors),
        weights: context.input('float32', input.weights.weights)
      },
      output: {
        coefficients: output.coefficients.view,
        covariance: output.covariance.view,
        summary: output.summary.view,
        fitted: output.fitted.view,
        residuals: output.residuals.view,
        status: status.view
      }
    });
    const nodes = contributor.getCommandNodes(context.graph);
    const record = getWGSLKernelNodeRecord(
      context.graph,
      nodes.find(node => node.id.endsWith('-pack'))!.id
    )!;
    context.graph.add(nodes);
    context.graph.add(
      createWGSLKernelNode(context.graph, {
        id: 'r5-read-within',
        operation: 'PanelReview',
        invocationCount: 12,
        bindings: [
          record.bindings.find(binding => binding.name === 'data')!,
          {name: 'output', view: within.view, type: 'f32', access: 'read_write'}
        ],
        body: 'output[outputOffset+index]=data[dataOffset+index*8u+1u];'
      })
    );
    context.run();
    expect(await readFloat32(within.buffer, 12)).toEqual(new Array(12).fill(0));
    expect(await readUint32(status.buffer, 1)).toEqual([2]);
    // A singular solve can retain its initial coefficients/summary as failure diagnostics.
    // Completed inference is withheld. A certified pole's all-NaN convention does not
    // apply to this singular-design failure.
    for (const name of ['covariance', 'fitted', 'residuals'] as const) {
      const port = output[name];
      const values = await readFloat32(port.buffer, port.view.length);
      expect(values.every(Number.isNaN), `${name}: ${JSON.stringify(values)}`).toBe(true);
    }
    for (const Operation of [
      GPUSpatialPanelLagMaximumLikelihoodOperation,
      CPUSpatialPanelLagMaximumLikelihoodOperation
    ]) {
      const result = await runOperationGraph(context.device, {
        inputs: {
          predictors: {format: 'float32', values: input.predictors},
          response: {format: 'float32', values: input.response},
          offsets: {format: 'uint32', values: input.weights.offsets},
          neighbors: {format: 'uint32', values: input.weights.neighbors},
          weights: {format: 'float32', values: input.weights.weights}
        },
        operations: [
          new Operation({
            entityCount: 4,
            periodCount: 3,
            predictorCount: 1,
            effects: 'unit',
            predictors: 'predictors',
            response: 'response',
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              summary: 'summary',
              fitted: 'fitted',
              residuals: 'residuals',
              status: 'status'
            }
          })
        ],
        outputs: ['coefficients', 'covariance', 'summary', 'fitted', 'residuals', 'status']
      });
      expect(Array.from(result['status'])).toEqual([2]);
      for (const name of ['covariance', 'fitted', 'residuals'])
        expect(Array.from(result[name]).every(Number.isNaN), name).toBe(true);
    }
  } finally {
    context.destroy();
  }
});

for (const tiled of [false, true])
  for (const extreme of [false, true])
    it(`R4 native identifiable within design survives pooled collinearity, tiled=${tiled}, extreme=${extreme}`, async () => {
      const context = await createRegressionTestContext(),
        input = extreme ? createExtremeOffsetPanelLagInput() : createOffsetPanelLagInput();
      try {
        const output = {
          coefficients: context.output('float32', 5),
          covariance: context.output('float32', 16),
          summary: context.output('float32', 6),
          fitted: context.output('float32', 16),
          residuals: context.output('float32', 16),
          status: context.output('uint32', 1)
        };
        context.graph.add(
          new GPUSpatialPanelLagMaximumLikelihood({
            effects: 'unit',
            periodCount: 4,
            predictorCount: 2,
            eigenvalues: tiled ? context.input('float32', [-1, 0, 0, 1]) : undefined,
            predictors: context.input('float32', input.predictors),
            response: context.input('float32', input.response),
            weights: {
              offsets: context.input('uint32', input.weights.offsets),
              neighbors: context.input('uint32', input.weights.neighbors),
              weights: context.input('float32', input.weights.weights)
            },
            output: {
              coefficients: output.coefficients.view,
              covariance: output.covariance.view,
              summary: output.summary.view,
              fitted: output.fitted.view,
              residuals: output.residuals.view,
              status: output.status.view
            }
          })
        );
        context.run();
        expectOffsetPanelLag(
          {
            coefficients: await readFloat32(output.coefficients.buffer, 5),
            covariance: await readFloat32(output.covariance.buffer, 16),
            summary: await readFloat32(output.summary.buffer, 6),
            fitted: await readFloat32(output.fitted.buffer, 16),
            residuals: await readFloat32(output.residuals.buffer, 16),
            status: await readUint32(output.status.buffer, 1)
          },
          extreme ? EXTREME_OFFSET_PANEL_LAG_EXPECTED : undefined
        );
      } finally {
        context.destroy();
      }
    });

for (const route of ['random', 'unit', 'tiled'] as const)
  for (const countPresent of [
    false,
    true,
    'spare',
    'inflated',
    'matching',
    'leadingValid',
    'leadingInvalid'
  ] as const)
    it(`R3 count 8/required 9 suppresses both operation forms: ${route}, count=${countPresent}`, async () => {
      const context = await createRegressionTestContext();
      const input = createPanelAuditInput();
      if (countPresent === 'spare' || countPresent === 'inflated' || countPresent === 'matching') {
        input.weights.neighbors = Uint32Array.from([...input.weights.neighbors, 0, 0]);
        input.weights.weights = Float32Array.from([...input.weights.weights, 0, 0]);
      }
      if (countPresent === 'leadingValid' || countPresent === 'leadingInvalid') {
        input.weights.offsets = Uint32Array.of(1, 3, 5, 7, 9);
        input.weights.neighbors = Uint32Array.of(0, ...input.weights.neighbors);
        input.weights.weights = Float32Array.of(0, ...input.weights.weights);
      }
      try {
        for (const Operation of [
          GPUSpatialPanelLagMaximumLikelihoodOperation,
          CPUSpatialPanelLagMaximumLikelihoodOperation
        ]) {
          const result = await runOperationGraph(context.device, {
            inputs: {
              predictors: {format: 'float32', values: input.predictors},
              response: {format: 'float32', values: input.response},
              offsets: {format: 'uint32', values: input.weights.offsets},
              neighbors: {format: 'uint32', values: input.weights.neighbors},
              weights: {format: 'float32', values: input.weights.weights},
              required: {
                format: 'uint32',
                values: Uint32Array.of(
                  countPresent === 'matching' || countPresent === 'leadingValid' ? 8 : 9
                )
              },
              ...(countPresent
                ? {
                    written: {
                      format: 'uint32' as const,
                      values: Uint32Array.of(
                        countPresent === 'inflated' || countPresent === 'leadingInvalid' ? 9 : 8
                      )
                    }
                  }
                : {}),
              ...(route === 'tiled'
                ? {spectrum: {format: 'float32' as const, values: Float32Array.of(-1, 0, 0, 1)}}
                : {})
            },
            operations: [
              new Operation({
                entityCount: 4,
                periodCount: 4,
                predictorCount: 1,
                effects: route === 'random' ? 'random' : 'unit',
                eigenvalues: route === 'tiled' ? 'spectrum' : undefined,
                predictors: 'predictors',
                response: 'response',
                weights: {
                  offsets: 'offsets',
                  neighbors: 'neighbors',
                  weights: 'weights',
                  status: {requiredCount: 'required', ...(countPresent ? {count: 'written'} : {})}
                },
                output: {
                  coefficients: 'coefficients',
                  covariance: 'covariance',
                  summary: 'summary',
                  fitted: 'fitted',
                  residuals: 'residuals',
                  status: 'status'
                },
                hausman:
                  route === 'random' ? {tests: 'hausman', status: 'hausmanStatus'} : undefined
              })
            ],
            outputs: [
              'coefficients',
              'covariance',
              'summary',
              'fitted',
              'residuals',
              'status',
              ...(route === 'random' ? ['hausman', 'hausmanStatus'] : [])
            ]
          });
          if (countPresent === 'matching' || countPresent === 'leadingValid') {
            const expected = PANEL_AUDIT_EXPECTED[route === 'random' ? 'random' : 'unit'];
            expect(Array.from(result['status'])).toEqual([0]);
            for (const port of ['coefficients', 'covariance', 'fitted', 'residuals'] as const)
              expectPanelAuditRows(result[port], expected[port], 5e-4);
            expectPanelAuditRows(Array.from(result['summary']).slice(0, 2), expected.summary, 5e-4);
            if (route === 'random') {
              expect(Array.from(result['hausmanStatus'])).toEqual([0]);
              expectPanelAuditRows(result['hausman'], PANEL_AUDIT_EXPECTED.random.hausman, 5e-4);
            }
            continue;
          }
          expect(Array.from(result['status'])).toEqual([1]);
          for (const name of [
            'coefficients',
            'covariance',
            'summary',
            'fitted',
            'residuals',
            ...(route === 'random' ? ['hausman'] : [])
          ]) {
            const expectedLength =
              name === 'coefficients'
                ? 4
                : name === 'covariance'
                  ? 9
                  : name === 'summary'
                    ? 6
                    : name === 'hausman'
                      ? 3
                      : 16;
            expect(result[name]).toHaveLength(expectedLength);
            expect(Array.from(result[name]).every(Number.isNaN)).toBe(true);
          }
          if (route === 'random') expect(Array.from(result['hausmanStatus'])).toEqual([1]);
        }
      } finally {
        context.destroy();
      }
    });

for (const sceneName of ['rationalSlopePole', 'weightedRationalPole'])
  for (const predictorScale of [1, 2 ** -25])
    for (const perturbed of [false, true])
      it(`W5-37 scale=${predictorScale} R1 native rational-slope certificate ${sceneName}, one-ULP control=${perturbed}`, async () => {
        const scene = panelPoleScenes.find(value => value.name === sceneName)!;
        const response = Float32Array.from(scene.response);
        if (perturbed) new Uint32Array(response.buffer)[0] += response[0] < 0 ? -1 : 1;
        const context = await createRegressionTestContext();
        try {
          const outputs = {
            coefficients: context.output('float32', 4),
            covariance: context.output('float32', 9),
            summary: context.output('float32', 6),
            fitted: context.output('float32', 8),
            residuals: context.output('float32', 8)
          };
          const status = context.output('uint32', 1);
          for (const [name, output] of Object.entries(outputs))
            context.graph.add(
              createWGSLKernelNode(context.graph, {
                id: `r1-seed-${name}`,
                operation: 'PanelReview',
                invocationCount: output.view.length,
                bindings: [{name: 'output', view: output.view, type: 'f32', access: 'read_write'}],
                declarations: GET_NAN_WGSL,
                body:
                  name === 'coefficients'
                    ? `let values=array<f32,4>(getNaN(),${1 / (3 * predictorScale)},${sceneName === 'weightedRationalPole' ? '0.5' : '0.0'},0.0);output[outputOffset+index]=values[index];`
                    : 'output[outputOffset+index]=987.0;'
              })
            );
          context.graph.add(
            getPanelLagPoleNodes(context.graph, 'r1-certificate', {
              periodCount: 2,
              predictorCount: 1,
              effects: 'unit',
              predictors: context.input(
                'float32',
                scene.predictors.map(value => value * predictorScale)
              ),
              response: context.input('float32', response),
              weights: {
                offsets: context.input('uint32', scene.weights.offsets),
                neighbors: context.input('uint32', scene.weights.neighbors),
                weights: context.input('float32', scene.weights.weights)
              },
              output: {
                coefficients: outputs.coefficients.view,
                covariance: outputs.covariance.view,
                summary: outputs.summary.view,
                fitted: outputs.fitted.view,
                residuals: outputs.residuals.view,
                status: status.view
              }
            })
          );
          context.run();
          expect(await readUint32(status.buffer, 1)).toEqual([perturbed ? 0 : 2]);
          for (const [name, output] of Object.entries(outputs)) {
            const values = await readFloat32(output.buffer, output.view.length);
            if (!perturbed) expect(values.every(Number.isNaN)).toBe(true);
            else if (name !== 'coefficients')
              expect(values).toEqual(new Array(output.view.length).fill(987));
          }
        } finally {
          context.destroy();
        }
      });

import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const model of ['panel', 'error'] as const)
  it(`W5-35 GPU positive certificate recovers fit and diagnostics ${model}`, async () => {
    const context = await createRegressionTestContext();
    const input = createPanelAuditInput();
    try {
      const complete = context.output('uint32', 1),
        table = context.output('float32', 12),
        summary = context.output('float32', 22),
        residuals = context.output('float32', 16),
        status = context.output('uint32', 1),
        tests = context.output('float32', 6),
        diagnosticStatus = context.output('uint32', 1);
      const props = {
        periodCount: 4,
        predictorCount: 1,
        effects: 'unit' as const,
        predictors: context.input('float32', input.predictors),
        response: context.input('float32', input.response),
        weights: {
          offsets: context.input('uint32', input.weights.offsets),
          neighbors: context.input('uint32', input.weights.neighbors),
          weights: context.input('float32', input.weights.weights),
          status: {complete: complete.view}
        },
        output: {
          table: table.view,
          summary: summary.view,
          residuals: residuals.view,
          status: {code: status.view}
        }
      };
      context.graph.add(
        model === 'panel'
          ? new GPUSpatialPanelRegression(props)
          : new GPUSpatialPanelErrorMaximumLikelihood({
              ...props,
              maximumPower: 8,
              diagnostics: {tests: tests.view, status: diagnosticStatus.view}
            })
      );
      const compiled = context.graph.compile();
      try {
        let first: number[] | undefined;
        for (const certificate of [1, 0, 1]) {
          complete.buffer.write(Uint32Array.of(certificate));
          submitGraph(context.device, compiled, undefined);
          expect(Array.from(await readUint32(status.buffer, 1))).toEqual([certificate ? 0 : 3]);
          const actual = await readFloat32(table.buffer, 12);
          if (certificate) {
            expect(Number.isFinite(actual[4])).toBe(true);
            if (model === 'panel')
              expectPanelAuditRows(actual, PANEL_REGRESSION_AUDIT_EXPECTED.unit.table, 5e-4);
            else
              expectPanelAuditRows(
                actual,
                [
                  NaN,
                  NaN,
                  NaN,
                  NaN,
                  1.4543553721878475,
                  0.16217494003990773,
                  8.96781815877325,
                  3.024458354593923e-19,
                  -0.27636131663120095,
                  0.22663902994434895,
                  -1.2193897789761157,
                  0.22269628745043568
                ],
                5e-4
              ); // NumPy/SciPy error-cycle.py, retained Q=8 objective.
            if (first) expectPanelAuditRows(actual, Array.from(first), 1e-6);
            else first = actual;
          } else expect(Array.from(actual).every(Number.isNaN)).toBe(true);
          if (model === 'error') {
            expect(Array.from(await readUint32(diagnosticStatus.buffer, 1))).toEqual([
              certificate ? 0 : 1
            ]);
            const values = await readFloat32(tests.buffer, 6);
            if (certificate) expectPanelAuditRows(values, PANEL_DIAGNOSTIC_AUDIT_EXPECTED, 5e-4);
            else expect(Array.from(values).every(Number.isNaN)).toBe(true);
          }
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      context.destroy();
    }
  });

it('CCK-03 GPU parallel pivot reduction retains a pivot in the last group', async () => {
  const context = await createRegressionTestContext();
  try {
    recordWGSLKernelNodes(context.graph);
    const rows = 8256,
      predictors = new Float32Array(rows);
    predictors[rows - 1] = 1;
    const nodes = getPanelLagPoleNodes(context.graph, 'last-group', {
      periodCount: 2064,
      predictorCount: 1,
      effects: 'unit',
      predictors: context.input('float32', predictors),
      response: context.input('float32', new Float32Array(rows)),
      weights: {
        offsets: context.input('uint32', [0, 1, 2, 3, 4]),
        neighbors: context.input('uint32', [1, 0, 3, 2]),
        weights: context.input('float32', [1, 1, 1, 1])
      },
      output: {
        coefficients: context.output('float32', 4).view,
        covariance: context.output('float32', 9).view,
        summary: context.output('float32', 6).view,
        status: context.output('uint32', 1).view,
        fitted: context.output('float32', rows).view,
        residuals: context.output('float32', rows).view
      }
    }).filter(node => node.id.includes('-pole-proportional-pivot'));
    context.graph.add(nodes);
    const pivotRecord = getWGSLKernelNodeRecord(context.graph, nodes[0].id)!;
    const pivots = pivotRecord.bindings.find(binding => binding.name === 'pivots')!.view;
    const output = context.output('uint32', 1);
    context.graph.add(
      createWGSLKernelNode(context.graph, {
        id: 'read-final-pivot',
        operation: 'PanelReview',
        invocationCount: 1,
        bindings: [
          {name: 'pivots', view: pivots, type: 'u32', access: 'read'},
          {name: 'output', view: output.view, type: 'u32', access: 'read_write'}
        ],
        body: 'output[outputOffset]=pivots[pivotsOffset+129u];'
      })
    );
    context.run();
    expect(Array.from(await readUint32(output.buffer, 1))).toEqual([rows - 1]);
  } finally {
    context.destroy();
  }
});

for (const incomingStatus of [0, 1])
  it(`CCK-02 GPU conditional diagnostic clear incoming=${incomingStatus}`, async () => {
    const context = await createRegressionTestContext();
    try {
      recordWGSLKernelNodes(context.graph);
      const tests = context.output('float32', 6),
        status = context.output('uint32', 1);
      tests.buffer.write(new Float32Array(6).fill(7));
      status.buffer.write(Uint32Array.of(incomingStatus));
      const nodes = new GPUSpatialPanelErrorMaximumLikelihood({
        periodCount: 4,
        predictorCount: 1,
        maximumPower: 3,
        predictors: context.input('float32', new Float32Array(16)),
        response: context.input('float32', new Float32Array(16)),
        weights: {
          offsets: context.input('uint32', [0, 1, 2, 3, 4]),
          neighbors: context.input('uint32', [1, 0, 3, 2]),
          weights: context.input('float32', [1, 1, 1, 1])
        },
        diagnostics: {tests: tests.view, status: status.view},
        output: {
          table: context.output('float32', 12).view,
          summary: context.output('float32', 22).view,
          status: {code: context.output('uint32', 1).view}
        }
      }).getCommandNodes(context.graph);
      const node = nodes.find(node => node.id.endsWith('-affine-reject-diagnostics'))!;
      const record = getWGSLKernelNodeRecord(context.graph, node.id)!;
      const flags = record.bindings.find(binding => binding.name === 'flags')!.view;
      context.graph.add(
        createWGSLKernelNode(context.graph, {
          id: 'seed-affine-flag',
          operation: 'PanelReview',
          invocationCount: 2,
          bindings: [{name: 'flags', view: flags, type: 'u32', access: 'read_write'}],
          body: 'flags[flagsOffset+index]=select(0u,1u,index==0u);'
        })
      );
      context.graph.add(node);
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([
        incomingStatus === 0 ? 2 : incomingStatus
      ]);
      const actual = Array.from(await readFloat32(tests.buffer, 6));
      expect(actual.every(incomingStatus === 0 ? Number.isNaN : value => value === 7)).toBe(true);
    } finally {
      context.destroy();
    }
  });

for (const predictorScale of [1, 2 ** -25])
  for (const nearby of [false, true])
    it(`W5-37 production GPU pole and identifiable control scale=${predictorScale} nearby=${nearby}`, async () => {
      const context = await createRegressionTestContext();
      try {
        const scene = panelPoleScenes.find(value => value.name === 'weightedRationalPole')!;
        const response = Float32Array.from(scene.response);
        if (nearby) response[0] += 0.125;
        const coefficients = context.output('float32', 4),
          summary = context.output('float32', 6),
          status = context.output('uint32', 1);
        context.graph.add(
          new GPUSpatialPanelLagMaximumLikelihood({
            periodCount: 2,
            predictorCount: 1,
            effects: 'unit',
            predictors: context.input(
              'float32',
              scene.predictors.map(value => value * predictorScale)
            ),
            response: context.input('float32', response),
            weights: {
              offsets: context.input('uint32', scene.weights.offsets),
              neighbors: context.input('uint32', scene.weights.neighbors),
              weights: context.input('float32', scene.weights.weights)
            },
            output: {
              coefficients: coefficients.view,
              summary: summary.view,
              covariance: context.output('float32', 9).view,
              fitted: context.output('float32', 8).view,
              residuals: context.output('float32', 8).view,
              status: status.view
            }
          })
        );
        context.run();
        expect(Array.from(await readUint32(status.buffer, 1))).toEqual([nearby ? 0 : 2]);
        const actual = await readFloat32(coefficients.buffer, 4),
          values = await readFloat32(summary.buffer, 6);
        if (nearby) {
          expect(Number.isFinite(actual[1])).toBe(true);
          expect(Number.isFinite(values[1])).toBe(true);
        } else {
          expect(Array.from(actual).every(Number.isNaN)).toBe(true);
          expect(Array.from(values).every(Number.isNaN)).toBe(true);
        }
      } finally {
        context.destroy();
      }
    });
