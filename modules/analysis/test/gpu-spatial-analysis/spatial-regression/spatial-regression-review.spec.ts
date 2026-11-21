// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialErrorGMOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm-operation';
import {GPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation';
import {GPUSpatialRegressionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics-operation';
import {GPUInstrumentalVariablesCovariance} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance';
import {getGPUSpatialTwoStageLeastSquaresModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-model-result';
import {
  runOperationGraph,
  makeCPUVector,
  getCPUVectorRows,
  type OperationGraphInput
} from '../../utils/mixed-execution-test-utils';
import {GPUGraph, gpuGraph} from '@luma.gl/graph/gpu-core';
import {CPUVector} from '@luma.gl/graph/gpu-data';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {createRegressionResidualDisplayNode} from '../../../../../examples/deck/spatial-analysis-shared/regression-residual-display';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import fixtures from './spatial-two-stage-least-squares-audit-fixtures.json';
import gmFixtures from './spatial-error-gm-audit-fixtures.json';
import {
  createReviewInput,
  createReviewKernel,
  createAffineReviewInput,
  createRationalAffineReviewInput,
  createConstantFittedReviewInput,
  createLongAffineReviewInput,
  createCycleReviewInput,
  createCompleteReviewInput,
  createSaturatedReviewInput
} from './spatial-regression-review-fixtures';
import {
  IV_OUTLIER_HC0,
  IV_OUTLIER_ROW_COUNT,
  createBipartiteReviewInput,
  createInstrumentalVariablesOutlierColumn,
  createTranslatedRationalAffineInput,
  createWeightedConstantFittedInput
} from './regression-diagnostics-w5-fixtures';

function graphInputs(
  input: ReturnType<typeof createReviewInput> & {residuals?: Float32Array},
  kernel?: ReturnType<typeof createReviewKernel>
): Record<string, OperationGraphInput> {
  return {
    predictors: {format: 'float32', values: input.predictors},
    response: {format: 'float32', values: input.response},
    offsets: {format: 'uint32', values: input.weights.offsets},
    neighbors: {format: 'uint32', values: input.weights.neighbors},
    weights: {format: 'float32', values: input.weights.weights},
    ...(input.residuals ? {residuals: {format: 'float32' as const, values: input.residuals}} : {}),
    ...(kernel
      ? {
          kernelOffsets: {format: 'uint32' as const, values: kernel.offsets},
          kernelNeighbors: {format: 'uint32' as const, values: kernel.neighbors},
          kernelWeights: {format: 'float32' as const, values: kernel.weights}
        }
      : {})
  };
}
const weightNames = {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'};
const fitOutput = {
  table: 'table',
  summary: 'summary',
  residuals: 'residuals',
  status: {code: 'code'}
};
const robustOutput = {covariance: 'covariance', status: {code: 'robustCode'}};

for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const)
  for (const departure of [false, true])
    it(`review 2 R3 GPU rational 1/7 slope ${estimator}, departure=${departure}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: graphInputs(createRationalAffineReviewInput(departure)),
        operations: [
          new GPUSpatialErrorGMOperation({
            rowCount: 12,
            predictorCount: 1,
            estimator,
            weights: weightNames,
            predictors: 'predictors',
            response: 'response',
            output: fitOutput
          })
        ],
        outputs: ['table', 'summary', 'code']
      });
      // Exact integer relation 7y=x; augmented rational rank is 2/3.
      expect(Array.from(result['code'])).toEqual([departure ? 0 : 3]);
      if (!departure) {
        expect(result['table'].every(Number.isNaN)).toBe(true);
        expect(result['summary'][0]).toBe(12);
        expect(result['summary'].subarray(1).every(Number.isNaN)).toBe(true);
      }
    });

it('review 2 R2 GPU constant fitted response keeps J=64 and robust LM undefined', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: graphInputs(createConstantFittedReviewInput()),
    operations: [
      new GPUSpatialRegressionDiagnosticsOperation({
        rowCount: 64,
        predictorCount: 1,
        weights: weightNames,
        predictors: 'predictors',
        response: 'response',
        residuals: 'residuals',
        output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
      })
    ],
    outputs: ['tests', 'summary', 'code']
  });
  expect(Array.from(result['code'])).toEqual([0]);
  expect(result['summary'][3]).toBe(64);
  for (const row of [0, 1]) expect(result['tests'][row * 3]).toBe(0);
  for (const row of [2, 3, 4]) expect(result['tests'][row * 3]).toBeNaN();
});

for (const [increment, variance] of [
  [2 ** -20, 8.7005227e-16],
  [1 / 32, 9.3212977e-7]
])
  it(`review 2 R7 GPU nearly complete graph retains tiny positive variance ${increment}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: graphInputs(createCompleteReviewInput(8, increment)),
      operations: [
        new GPUSpatialRegressionDiagnosticsOperation({
          rowCount: 8,
          predictorCount: 1,
          weights: weightNames,
          predictors: 'predictors',
          response: 'response',
          residuals: 'residuals',
          output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
        })
      ],
      outputs: ['tests', 'summary', 'code']
    });
    expect(Array.from(result['code'])).toEqual([0]);
    // 0.1% budget for f32 projection and norm reduction, including variance near 10^-15.
    expect(Number.isFinite(result['summary'][7])).toBe(true);
    expect(Math.abs(result['summary'][7] / variance - 1)).toBeLessThan(1e-3);
    expect(result['summary'][8]).toBeCloseTo(1.15470054, 3);
    expect(result['summary'][9]).toBeCloseTo(0.24821308, 3);
    expect(result['tests'][15]).toBeCloseTo(1.15470054, 3);
    expect(result['tests'][17]).toBeCloseTo(0.24821308, 3);
  });

it('review 2 R5 GPU one-ULP nonconstant IV column has rank and zero HC0 covariance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const column = new Float32Array(512).fill(Math.fround(1.9793816));
  column[511] += 2 ** -23;
  const result = await runOperationGraph(device, {
    inputs: {
      column: {format: 'float32', values: column},
      residuals: {format: 'float32', values: new Float32Array(512)}
    },
    operations: [
      new GPUContributorOperation({
        id: 'iv-one-ulp',
        inputs: {column: 'column', residuals: 'residuals'},
        outputs: {
          covariance: {value: 'covariance', format: 'float32', length: 4},
          code: {value: 'code', format: 'uint32', length: 1}
        },
        createContributor: getView =>
          new GPUInstrumentalVariablesCovariance({
            design: getView<'float32'>('column'),
            instruments: getView<'float32'>('column'),
            residuals: getView<'float32'>('residuals'),
            coefficientCount: 1,
            instrumentCount: 1,
            method: 'HC0',
            output: {
              covariance: getView<'float32'>('covariance'),
              status: {code: getView<'uint32'>('code')}
            }
          })
      })
    ],
    outputs: ['covariance', 'code']
  });
  // True centered SS = (511/512)*2^-46 >0; full-rank identical X/H and zero u imply V=0.
  expect(Array.from(result['code'])).toEqual([0]);
  expect(Array.from(result['covariance'])).toEqual([0, 0, 0, 0]);
});

it('review 2 R11 GPU example display hides failed finite residuals and restores the next submission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const operation = new GPUContributorOperation({
    id: 'display-test',
    inputs: {residuals: 'residuals', code: 'code'},
    outputs: {display: {value: 'display', format: 'float32', length: 3}},
    createContributor: getView => ({
      getCommandNodes: graph => [
        createRegressionResidualDisplayNode(graph, {
          id: 'display-guard',
          residuals: getView<'float32'>('residuals'),
          code: getView<'uint32'>('code'),
          output: getView<'float32'>('display')
        })
      ]
    })
  });
  const graph = new GPUGraph();
  const residuals = makeCPUVector('float32', Float32Array.of(0, -0.125, 1.25));
  const code = makeCPUVector('uint32', Uint32Array.of(2));
  graph.addInput('residuals', gpuGraph.vector('float32', {length: 3}));
  graph.addInput('code', gpuGraph.vector('uint32', {length: 1}));
  graph.add(operation);
  const compiled = graph.compile(device, {
    outputs: ['display'],
    outputLocations: {display: 'cpu'},
    lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
  });
  try {
    const failed = (await compiled.run({inputs: {residuals, code}})) as Record<string, unknown>;
    expect(failed['display']).toBeInstanceOf(CPUVector);
    if (!(failed['display'] instanceof CPUVector)) throw new Error('display output must be on CPU');
    expect(getCPUVectorRows(failed['display']).every(Number.isNaN)).toBe(true);
    const successful = (await compiled.run({
      inputs: {residuals, code: makeCPUVector('uint32', Uint32Array.of(0))}
    })) as Record<string, unknown>;
    expect(successful['display']).toBeInstanceOf(CPUVector);
    if (!(successful['display'] instanceof CPUVector))
      throw new Error('display output must be on CPU');
    expect(Array.from(getCPUVectorRows(successful['display']))).toEqual([0, -0.125, 1.25]);
  } finally {
    compiled.destroy();
    operation.destroy();
  }
});

it('R1 robust 2SLS spare residual capacity preserves the independent HC0 covariance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = createReviewInput();
  const result = await runOperationGraph(device, {
    inputs: graphInputs(input),
    operations: [
      new GPUContributorOperation({
        id: 'spare-residuals',
        inputs: {predictors: 'predictors', response: 'response', ...weightNames},
        outputs: {
          table: {value: 'table', format: 'float32', length: 16},
          summary: {value: 'summary', format: 'float32', length: 16},
          residuals: {value: 'residuals', format: 'float32', length: 13},
          code: {value: 'code', format: 'uint32', length: 1},
          covariance: {value: 'covariance', format: 'float32', length: 16},
          robustCode: {value: 'robustCode', format: 'uint32', length: 1}
        },
        createContributor: getView =>
          new GPUSpatialTwoStageLeastSquares({
            predictorCount: 2,
            predictors: getView<'float32'>('predictors'),
            response: getView<'float32'>('response'),
            weights: {
              offsets: getView<'uint32'>('offsets'),
              neighbors: getView<'uint32'>('neighbors'),
              weights: getView<'float32'>('weights')
            },
            robust: {
              method: 'HC0',
              output: {
                covariance: getView<'float32'>('covariance'),
                status: {code: getView<'uint32'>('robustCode')}
              }
            },
            output: {
              table: getView<'float32'>('table'),
              summary: getView<'float32'>('summary'),
              residuals: getView<'float32'>('residuals'),
              status: {code: getView<'uint32'>('code')}
            }
          })
      })
    ],
    outputs: ['table', 'summary', 'residuals', 'code', 'covariance', 'robustCode']
  });
  expect(Array.from(result['code'])).toEqual([0]);
  expect(Array.from(result['robustCode'])).toEqual([0]);
  // 0.5% float32 IV projection/sandwich budget, as in the ordinary independent audit.
  expectRegressionRowsClose(result['covariance'], fixtures[0].robust.HC0.covariance, 5e-3);
  expectRegressionRowsClose(result['table'], fixtures[0].robust.HC0.table, 5e-3);
  expectRegressionRowsClose(
    result['summary'],
    fixtures[0].summary.map(value => value ?? NaN),
    5e-3
  );
  expectRegressionRowsClose(result['residuals'].subarray(0, 12), fixtures[0].residuals, 5e-3);
});

for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const)
  for (const departure of [false, true]) {
    it(`R3 GPU GM ${estimator} exact affine=${!departure} retains the identification boundary`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = createAffineReviewInput(departure);
      const result = await runOperationGraph(device, {
        inputs: graphInputs(input),
        operations: [
          new GPUSpatialErrorGMOperation({
            rowCount: 12,
            predictorCount: 2,
            estimator,
            weights: weightNames,
            predictors: 'predictors',
            response: 'response',
            output: fitOutput
          })
        ],
        outputs: ['table', 'summary', 'residuals', 'code']
      });
      expect(Array.from(result['code'])).toEqual([departure ? 0 : 3]);
      if (!departure) {
        expect(result['table'].every(Number.isNaN)).toBe(true);
        expect(result['summary'][0]).toBe(12);
        expect(result['summary'].subarray(1).every(Number.isNaN)).toBe(true);
      } else {
        expect(Number.isFinite(result['summary'][1])).toBe(true);
        expect(result['summary'][4]).toBeGreaterThan(0);
      }
    });
  }

for (const departure of [false, true])
  it(`R3 GPU affine certificate verifies all 300 rows, unsampled departure=${departure}`, async () => {
    // review.oracle.py: rank([1,X,y]) is 3 for exact, 4 for a 2^-23 departure at row 6.
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: graphInputs(createLongAffineReviewInput(departure)),
      operations: [
        new GPUSpatialErrorGMOperation({
          rowCount: 300,
          predictorCount: 2,
          estimator: 'kelejian-prucha',
          weights: weightNames,
          predictors: 'predictors',
          response: 'response',
          output: fitOutput
        })
      ],
      outputs: ['table', 'summary', 'residuals', 'code']
    });
    expect(Array.from(result['code'])).toEqual([departure ? 0 : 3]);
    if (!departure) {
      expect(result['table'].every(Number.isNaN)).toBe(true);
      expect(result['summary'][0]).toBe(300);
      expect(result['summary'].subarray(1).every(Number.isNaN)).toBe(true);
    } else expect(result['summary'][4]).toBeGreaterThan(0);
  });

it('R4 GPU saturated first stage has a valid fit, unit partial R2 and undefined F', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: graphInputs(createSaturatedReviewInput()),
    operations: [
      new GPUSpatialTwoStageLeastSquaresOperation({
        rowCount: 5,
        predictorCount: 2,
        weights: weightNames,
        predictors: 'predictors',
        response: 'response',
        output: fitOutput
      })
    ],
    outputs: ['table', 'summary', 'code']
  });
  expect(Array.from(result['code'])).toEqual([0]);
  expect(result['summary'][14]).toBeNaN();
  expect(result['summary'][15]).toBeCloseTo(1, 5);
  // Independent dense SVD projector pins from spatial-regression-review.oracle.py.
  expectRegressionRowsClose(
    [result['table'][0], result['table'][4], result['table'][8], result['table'][12]],
    [18.111798433012765, 0.179824289302262, -0.877199635770533, -0.5837687743278274],
    5e-3
  );
});

it('R5 constant float32 0.1 instrument is singular even with a zero sandwich', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      design: {format: 'float32', values: new Float32Array(8).fill(0.1)},
      residuals: {format: 'float32', values: new Float32Array(8)}
    },
    operations: [
      new GPUContributorOperation({
        id: 'constant-instrument-review',
        inputs: {design: 'design', residuals: 'residuals'},
        outputs: {
          code: {value: 'code', format: 'uint32', length: 1},
          covariance: {value: 'covariance', format: 'float32', length: 4}
        },
        createContributor: getView =>
          new GPUInstrumentalVariablesCovariance({
            coefficientCount: 1,
            instrumentCount: 1,
            design: getView<'float32'>('design'),
            instruments: getView<'float32'>('design'),
            residuals: getView<'float32'>('residuals'),
            output: {
              covariance: getView<'float32'>('covariance'),
              status: {code: getView<'uint32'>('code')}
            }
          })
      })
    ],
    outputs: ['code', 'covariance']
  });
  expect(Array.from(result['code'])).toEqual([1]);
  expect(result['covariance'].every(Number.isNaN)).toBe(true);
});

for (const loss of [0, 1])
  it(`R6 GPU robust readback preserves method and inference code with loss=${loss}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: {
        ...graphInputs(createReviewInput(), createReviewKernel()),
        loss: {format: 'uint32', values: Uint32Array.of(loss)}
      },
      operations: [
        new GPUSpatialTwoStageLeastSquaresOperation({
          rowCount: 12,
          predictorCount: 2,
          weights: weightNames,
          predictors: 'predictors',
          response: 'response',
          output: fitOutput,
          robust: {
            method: 'HAC',
            kernel: {
              offsets: 'kernelOffsets',
              neighbors: 'kernelNeighbors',
              weights: 'kernelWeights',
              status: {candidateOverflow: 'loss'}
            },
            output: robustOutput
          }
        })
      ],
      outputs: ['table', 'summary', 'code', 'robustCode']
    });
    const decoded = getGPUSpatialTwoStageLeastSquaresModelResult({
      table: Float32Array.from(result['table']),
      summary: Float32Array.from(result['summary']),
      status: Uint32Array.from(result['code']),
      robust: {method: 'HAC', status: Uint32Array.from(result['robustCode'])}
    });
    expect(decoded.status.ok).toBe(true);
    expect(decoded.standardErrorSigmaSquared).toBe('robust');
    expect(decoded.covarianceInference).toEqual({
      method: 'HAC',
      status: {code: loss ? 4 : 0, ok: !loss}
    });
  });

for (const offset of [0, 4096])
  it(`R2 GPU LM projection norm survives response offset ${offset}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: graphInputs(createCycleReviewInput(16, offset)),
      operations: [
        new GPUSpatialRegressionDiagnosticsOperation({
          rowCount: 16,
          predictorCount: 1,
          weights: weightNames,
          predictors: 'predictors',
          response: 'response',
          residuals: 'residuals',
          output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
        })
      ],
      outputs: ['tests', 'summary', 'code']
    });
    expect(Array.from(result['code'])).toEqual([0]);
    expect(result['summary'][3]).toBeCloseTo(32, 3);
    for (const [row, value] of [8, 0, 16, 8, 16].entries())
      expect(result['tests'][row * 3]).toBeCloseTo(value, 3);
  });

for (const count of [8, 12])
  it(`R7 GPU complete ${count}-row graph has zero Moran variance`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: graphInputs(createCompleteReviewInput(count)),
      operations: [
        new GPUSpatialRegressionDiagnosticsOperation({
          rowCount: count,
          predictorCount: 1,
          weights: weightNames,
          predictors: 'predictors',
          response: 'response',
          residuals: 'residuals',
          output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
        })
      ],
      outputs: ['tests', 'summary', 'code']
    });
    expect(Array.from(result['code'])).toEqual([0]);
    expect(result['summary'][7]).toBe(0);
    expect(result['summary'][5]).toBeCloseTo(-1 / (count - 1), 5);
    for (const slot of [8, 9]) expect(result['summary'][slot]).toBeNaN();
    for (const slot of [15, 17]) expect(result['tests'][slot]).toBeNaN();
  });

for (const exponent of [18, 29, 30])
  it(`R8/R12 GPU homoskedastic GM is equivariant at response scale 2^${exponent}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createReviewInput();
    input.response = Float32Array.from(input.response, value => value * 2 ** exponent);
    const result = await runOperationGraph(device, {
      inputs: graphInputs(input),
      operations: [
        new GPUSpatialErrorGMOperation({
          rowCount: 12,
          predictorCount: 2,
          estimator: 'homoskedastic',
          weights: weightNames,
          predictors: 'predictors',
          response: 'response',
          output: fitOutput
        })
      ],
      outputs: ['table', 'summary', 'code']
    });
    expect(Array.from(result['code'])).toEqual([0]);
    expect(result['table'][12]).toBeCloseTo(-0.9621903664806921, 3);
    expect(result['table'][13]).toBeCloseTo(0.5414640317274417, 3);
    expectRegressionRowsClose(
      [result['summary'][2] / 2 ** (2 * exponent)],
      [gmFixtures[0].expected.summary[2]!],
      5e-3
    );
    expect(result['summary'][3]).toBeCloseTo(0.7094877554580391, 3);
  });

it('R10 GPU HAC rejects nonzero initial CSR offset and preserves point estimates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: graphInputs(createReviewInput(), createReviewKernel(true)),
    operations: [
      new GPUSpatialTwoStageLeastSquaresOperation({
        rowCount: 12,
        predictorCount: 2,
        weights: weightNames,
        predictors: 'predictors',
        response: 'response',
        output: fitOutput,
        robust: {
          method: 'HAC',
          kernel: {
            offsets: 'kernelOffsets',
            neighbors: 'kernelNeighbors',
            weights: 'kernelWeights'
          },
          output: robustOutput
        }
      })
    ],
    outputs: ['table', 'summary', 'residuals', 'code', 'covariance', 'robustCode']
  });
  expect(Array.from(result['code'])).toEqual([0]);
  expect(Array.from(result['robustCode'])).toEqual([4]);
  expect(result['covariance'].every(Number.isNaN)).toBe(true);
  for (let row = 0; row < 4; row++) {
    expectRegressionRowsClose(
      result['table'].subarray(4 * row, 4 * row + 1),
      [fixtures[0].table[4 * row]],
      5e-3
    );
    expect(result['table'].subarray(4 * row + 1, 4 * row + 4).every(Number.isNaN)).toBe(true);
  }
  expectRegressionRowsClose(
    result['summary'],
    fixtures[0].summary.map(value => value ?? NaN),
    5e-3
  );
  expectRegressionRowsClose(result['residuals'], fixtures[0].residuals, 5e-3);
});

function runReviewDiagnostics(
  device: Parameters<typeof runOperationGraph>[0],
  input: ReturnType<typeof createBipartiteReviewInput>
) {
  return runOperationGraph(device, {
    inputs: graphInputs(input),
    operations: [
      new GPUSpatialRegressionDiagnosticsOperation({
        rowCount: input.response.length,
        predictorCount: 1,
        weights: weightNames,
        predictors: 'predictors',
        response: 'response',
        residuals: 'residuals',
        output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
      })
    ],
    outputs: ['tests', 'summary', 'code']
  });
}

it('W5-30 GPU K(8,8) partition design publishes zero Moran variance and undefined z/p', async () => {
  // W = ab' + ba' with a, b in the design span: MWM = 0, so Var(I) = 0 and I = 0 exactly.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runReviewDiagnostics(device, createBipartiteReviewInput());
  expect(Array.from(result['code'])).toEqual([0]);
  expect(result['summary'][5]).toBe(0);
  expect(result['summary'][7]).toBe(0);
  for (const slot of [8, 9]) expect(result['summary'][slot]).toBeNaN();
  for (const slot of [15, 17]) expect(result['tests'][slot]).toBeNaN();
});

// NumPy float64 reference (k88_reference.py in the stream scratch folder): z = 1.06904497.
it('W5-30 GPU nearly bipartite control keeps its positive variance at increment 1/32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runReviewDiagnostics(device, createBipartiteReviewInput(1 / 32));
  expect(Array.from(result['code'])).toEqual([0]);
  expect(Number.isFinite(result['summary'][7])).toBe(true);
  expect(Math.abs(result['summary'][7] / 5.2128607649655715e-8 - 1)).toBeLessThan(1e-2);
  expect(Number.isFinite(result['tests'][15])).toBe(true);
  expect(Math.abs(result['tests'][15] - 1.06904497)).toBeLessThan(1e-2);
});

it('W5-30 GPU nearly bipartite control at increment 2^-20 stays positive, not vanished', async () => {
  // True Var = 4.857e-17. The within-partition blocks of B = S - c11' + cI are O(1) and are
  // removed only by the float32 projection, so Metal measured 1.24e-16 (ratio 2.56): a resolved
  // nonzero operator, distinguished from the exactly vanished one above, not a precise variance.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runReviewDiagnostics(device, createBipartiteReviewInput(2 ** -20));
  expect(Array.from(result['code'])).toEqual([0]);
  expect(Number.isFinite(result['summary'][7])).toBe(true);
  expect(result['summary'][7]).toBeGreaterThan(0);
  expect(Number.isFinite(result['tests'][15])).toBe(true);
});

it('W5-31 GPU weighted constant fitted lag keeps J=T=35 and robust LM undefined', async () => {
  // Every row sums to 1, so W fitted is exactly constant: J = T = tr(W'W) + tr(W^2) = 33 + 2.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runReviewDiagnostics(device, createWeightedConstantFittedInput());
  expect(Array.from(result['code'])).toEqual([0]);
  expect(Math.abs(result['summary'][2] - 35)).toBeLessThan(1e-4);
  expect(Math.abs(result['summary'][3] - 35)).toBeLessThan(1e-4);
  for (const row of [2, 3, 4]) expect(result['tests'][row * 3]).toBeNaN();
});

for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const)
  for (const departure of [false, true])
    it(`W5-32 GPU GM ${estimator} translated 7y = x + 16777217, departure=${departure}`, async () => {
      // Exact integer identity; the departure moves y[0] by one ULP (0.25). CPU: status 3 / 0.
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: graphInputs(createTranslatedRationalAffineInput(departure)),
        operations: [
          new GPUSpatialErrorGMOperation({
            rowCount: 12,
            predictorCount: 1,
            estimator,
            weights: weightNames,
            predictors: 'predictors',
            response: 'response',
            output: fitOutput
          })
        ],
        outputs: ['table', 'summary', 'code']
      });
      expect(Array.from(result['code'])).toEqual([departure ? 0 : 3]);
      if (!departure) {
        expect(result['table'].every(Number.isNaN)).toBe(true);
        expect(result['summary'][0]).toBe(12);
      }
    });

for (const [outlierRow, constant] of [
  [0, false],
  [IV_OUTLIER_ROW_COUNT - 1, false],
  [0, true]
] as const)
  it(`W5-33 GPU IV rank uses centered moments, outlier row ${outlierRow}, constant=${constant}`, async () => {
    // Centered SS = v^2 (n-1)/n = 3.9179439 > 0 wherever the zero sits; a constant column is
    // singular. The HC0 sandwich for one unit residual at the outlier pins the centered moments.
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const column = constant
      ? new Float32Array(IV_OUTLIER_ROW_COUNT).fill(Math.fround(1.9793816))
      : createInstrumentalVariablesOutlierColumn(outlierRow);
    const residuals = new Float32Array(IV_OUTLIER_ROW_COUNT);
    residuals[outlierRow] = 1;
    const result = await runOperationGraph(device, {
      inputs: {
        column: {format: 'float32', values: column},
        residuals: {format: 'float32', values: residuals}
      },
      operations: [
        new GPUContributorOperation({
          id: 'iv-outlier',
          inputs: {column: 'column', residuals: 'residuals'},
          outputs: {
            covariance: {value: 'covariance', format: 'float32', length: 4},
            code: {value: 'code', format: 'uint32', length: 1}
          },
          createContributor: getView =>
            new GPUInstrumentalVariablesCovariance({
              design: getView<'float32'>('column'),
              instruments: getView<'float32'>('column'),
              residuals: getView<'float32'>('residuals'),
              coefficientCount: 1,
              instrumentCount: 1,
              method: 'HC0',
              output: {
                covariance: getView<'float32'>('covariance'),
                status: {code: getView<'uint32'>('code')}
              }
            })
        })
      ],
      outputs: ['covariance', 'code']
    });
    expect(Array.from(result['code'])).toEqual([constant ? 1 : 0]);
    if (constant) expect(Array.from(result['covariance']).every(Number.isNaN)).toBe(true);
    else
      for (const [index, expected] of IV_OUTLIER_HC0.entries()) {
        expect(Number.isFinite(result['covariance'][index])).toBe(true);
        expect(Math.abs(result['covariance'][index] - expected)).toBeLessThan(2e-3);
      }
  });
