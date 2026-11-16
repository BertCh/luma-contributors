// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation';
import {GPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import spatialFixture from './seemingly-unrelated-regression-spatial-operation-fixtures.json';

/** Orthogonal, repeated eight-row design with exactly known FGLS coefficients and covariance. */
function createScene(rows: number, predictorOffset = 0) {
  const design = [-3, -3, -1, -1, 1, 1, 3, 3];
  const firstErrors = [1, -1, 1, -1, 1, -1, 1, -1];
  const secondErrors = [1, -1, -1, 1, -1, 1, 1, -1];
  const spread = predictorOffset === 0 ? 1 : 32;
  const predictors = new Float32Array(rows * 2);
  const responses = new Float32Array(rows * 2);
  const initialResiduals = new Float32Array(rows * 2);
  for (let row = 0; row < rows; row++) {
    const centeredPredictor = spread * design[row % 8];
    predictors[row] = predictorOffset + centeredPredictor;
    predictors[rows + row] = predictorOffset + centeredPredictor;
    responses[row] = 3 + 0.5 * centeredPredictor + firstErrors[row % 8];
    responses[rows + row] = -5 - 0.25 * centeredPredictor + secondErrors[row % 8];
    initialResiduals[row] = firstErrors[row % 8];
    initialResiduals[rows + row] = secondErrors[row % 8];
  }
  return {predictors, responses, initialResiduals};
}

for (const [rows, offset] of [
  [128, 0],
  [1048576, 0],
  [128, 2 ** 27]
] as const) {
  it(`SUR preserves independent closed-form outputs at ${rows} rows and predictor offset ${offset}`, {
    timeout: 120000
  }, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = createScene(rows, offset);
    const run = (
      Operation:
        | typeof GPUSeeminglyUnrelatedRegressionOperation
        | typeof CPUSeeminglyUnrelatedRegressionOperation
    ) =>
      runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: scene.predictors},
          responses: {format: 'float32', values: scene.responses},
          initialResiduals: {format: 'float32', values: scene.initialResiduals}
        },
        operations: [
          new Operation({
            rowCount: rows,
            predictorCount: 1,
            equationCount: 2,
            predictors: 'predictors',
            responses: 'responses',
            initialResiduals: offset ? 'initialResiduals' : undefined,
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              residualCovariance: 'residualCovariance',
              residuals: 'residuals',
              status: 'status'
            }
          })
        ],
        outputs: ['coefficients', 'covariance', 'residualCovariance', 'residuals', 'status']
      });
    const actual = await run(GPUSeeminglyUnrelatedRegressionOperation);
    expect(Array.from(actual['status'])).toEqual([0]);
    expect(actual['coefficients'][1]).toBeCloseTo(0.5, 5);
    expect(actual['coefficients'][3]).toBeCloseTo(-0.25, 5);
    let maximumResidualError = 0;
    for (let index = 0; index < rows * 2; index++) {
      maximumResidualError = Math.max(
        maximumResidualError,
        Math.abs(actual['residuals'][index] - scene.initialResiduals[index])
      );
    }
    // The former raw-intercept residual reconstruction loses order-one values at offset 2^27.
    expect(maximumResidualError).toBeLessThan(1e-3);
    if (offset === 0) {
      expect(actual['coefficients'][0]).toBeCloseTo(3, 4);
      expect(actual['coefficients'][2]).toBeCloseTo(-5, 4);
      for (const [index, expected] of [1, 0, 0, 1].entries())
        expect(actual['residualCovariance'][index]).toBeCloseTo(expected, 4);
      for (let index = 0; index < 16; index++) {
        const expected = index === 0 || index === 10 ? 1 : index === 5 || index === 15 ? 0.2 : 0;
        expect(actual['covariance'][index] * rows).toBeCloseTo(expected, 4);
      }
    }
    if (rows === 128) {
      const reference = await run(CPUSeeminglyUnrelatedRegressionOperation);
      expect(Array.from(reference['status'])).toEqual([0]);
      let maximumDifference = 0;
      for (let index = 0; index < rows * 2; index++)
        maximumDifference = Math.max(
          maximumDifference,
          Math.abs(actual['residuals'][index] - reference['residuals'][index])
        );
      expect(maximumDifference).toBeLessThan(1e-3);
    }
  });
}

for (const rows of [65537, 1000001]) {
  it(`SUR distinct designs, correlated errors, masks and an uneven tail at ${rows} rows match the CPU oracle`, {
    timeout: 120000
  }, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const predictors = new Float32Array(rows * 2);
    const responses = new Float32Array(rows * 2);
    const mask = new Uint32Array(rows);
    for (let row = 0; row < rows; row++) {
      const first = Math.sin(row * 0.17);
      const independent = Math.cos(row * 0.31);
      predictors[row] = 10 + Math.cos(row * 0.11);
      predictors[rows + row] = -5 + Math.sin(row * 0.23);
      responses[row] = 2 + 0.75 * predictors[row] + first;
      responses[rows + row] = -3 - 0.5 * predictors[rows + row] + 0.6 * first + 0.8 * independent;
      mask[row] = row % 19 === 0 ? 0 : 1;
    }
    const run = (
      Operation:
        | typeof GPUSeeminglyUnrelatedRegressionOperation
        | typeof CPUSeeminglyUnrelatedRegressionOperation
    ) =>
      runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: predictors},
          responses: {format: 'float32', values: responses},
          mask: {format: 'uint32', values: mask}
        },
        operations: [
          new Operation({
            rowCount: rows,
            predictorCount: 1,
            equationCount: 2,
            predictors: 'predictors',
            responses: 'responses',
            mask: 'mask',
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              residualCovariance: 'residualCovariance',
              residuals: 'residuals',
              status: 'status'
            }
          })
        ],
        outputs: ['coefficients', 'covariance', 'residualCovariance', 'residuals', 'status']
      });
    const actual = await run(GPUSeeminglyUnrelatedRegressionOperation);
    const expected = await run(CPUSeeminglyUnrelatedRegressionOperation);
    expect(Array.from(actual['status'])).toEqual([0]);
    expect(Array.from(expected['status'])).toEqual([0]);
    for (const name of ['coefficients', 'residualCovariance', 'covariance']) {
      for (let index = 0; index < actual[name].length; index++) {
        const scale = name === 'covariance' ? rows : 1;
        expect(Math.abs(actual[name][index] - expected[name][index]) * scale).toBeLessThan(
          0.004 * Math.max(1, Math.abs(expected[name][index]) * scale)
        );
      }
    }
    let maximumError = 0;
    for (let index = 0; index < rows * 2; index++) {
      if (mask[index % rows] === 0) expect(actual['residuals'][index]).toBeNaN();
      else
        maximumError = Math.max(
          maximumError,
          Math.abs(actual['residuals'][index] - expected['residuals'][index])
        );
    }
    expect(maximumError).toBeLessThan(0.002);
  });
}

for (const failure of ['singular-design', 'singular-error'] as const) {
  it(`SUR publishes NaN residuals for ${failure}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = 129;
    const scene = createScene(136);
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {
          format: 'float32',
          values:
            failure === 'singular-design'
              ? new Float32Array(rows * 2).fill(1)
              : Float32Array.from({length: rows * 2}, (_, index) => Math.sin(index % rows))
        },
        responses: {format: 'float32', values: scene.responses.slice(0, rows * 2)},
        initial: {format: 'float32', values: new Float32Array(rows * 2).fill(1)}
      },
      operations: [
        new GPUSeeminglyUnrelatedRegressionOperation({
          rowCount: rows,
          predictorCount: 1,
          equationCount: 2,
          predictors: 'predictors',
          responses: 'responses',
          initialResiduals: failure === 'singular-error' ? 'initial' : undefined,
          output: {
            coefficients: 'coefficients',
            covariance: 'covariance',
            residualCovariance: 'residualCovariance',
            residuals: 'residuals',
            status: 'status'
          }
        })
      ],
      outputs: ['status', 'residuals']
    });
    expect(Array.from(result['status'])).toEqual([failure === 'singular-design' ? 2 : 3]);
    expect(result['residuals'].every(Number.isNaN)).toBe(true);
  });
}

it('spatial SUR above 65536 rows matches the independent CPU fit', {timeout: 120000}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Disjoint copies preserve valid sorted zero-diagonal CSR and a well-conditioned known scene.
  const copies = 1640;
  const baseRows = spatialFixture.rowCount;
  const rows = copies * baseRows;
  const columns = spatialFixture.predictorCount;
  const baseEdges = spatialFixture.weights.neighbors.length;
  const offsets = new Uint32Array(rows + 1);
  const neighbors = new Uint32Array(copies * baseEdges);
  const weights = new Float32Array(neighbors.length);
  const predictors = new Float32Array(rows * columns * 2);
  const responses = new Float32Array(rows * 2);
  for (let copy = 0; copy < copies; copy++) {
    for (let row = 0; row < baseRows; row++)
      offsets[copy * baseRows + row] = copy * baseEdges + spatialFixture.weights.offsets[row];
    for (let edge = 0; edge < baseEdges; edge++) {
      neighbors[copy * baseEdges + edge] = copy * baseRows + spatialFixture.weights.neighbors[edge];
      weights[copy * baseEdges + edge] = spatialFixture.weights.weights[edge];
    }
    for (let equation = 0; equation < 2; equation++) {
      predictors.set(
        spatialFixture.predictors.slice(
          equation * baseRows * columns,
          (equation + 1) * baseRows * columns
        ),
        (equation * rows + copy * baseRows) * columns
      );
      responses.set(
        spatialFixture.responses.slice(equation * baseRows, (equation + 1) * baseRows),
        equation * rows + copy * baseRows
      );
    }
  }
  offsets[rows] = neighbors.length;
  const run = (
    Operation:
      | typeof GPUSeeminglyUnrelatedRegressionOperation
      | typeof CPUSeeminglyUnrelatedRegressionOperation
  ) =>
    runOperationGraph(device, {
      inputs: {
        predictors: {format: 'float32', values: predictors},
        responses: {format: 'float32', values: responses},
        offsets: {format: 'uint32', values: offsets},
        neighbors: {format: 'uint32', values: neighbors},
        weights: {format: 'float32', values: weights}
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
      outputs: [
        'coefficients',
        'covariance',
        'residualCovariance',
        'residuals',
        'parameters',
        'spatialStatus',
        'status'
      ]
    });
  const actual = await run(GPUSeeminglyUnrelatedRegressionOperation);
  const expected = await run(CPUSeeminglyUnrelatedRegressionOperation);
  expect(Array.from(actual['status'])).toEqual([0]);
  expect(Array.from(expected['status'])).toEqual([0]);
  expect(Array.from(actual['spatialStatus'])).toEqual([0, 0]);
  expect(Array.from(expected['spatialStatus'])).toEqual([0, 0]);
  for (const name of [
    'coefficients',
    'covariance',
    'residualCovariance',
    'residuals',
    'parameters'
  ]) {
    for (let index = 0; index < actual[name].length; index++) {
      const scale = name === 'covariance' ? rows : 1;
      expect(Math.abs(actual[name][index] - expected[name][index]) * scale).toBeLessThan(
        0.01 * Math.max(1, Math.abs(expected[name][index]) * scale)
      );
    }
  }
});
