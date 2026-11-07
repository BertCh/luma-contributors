// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUInstrumentalVariablesCovariance of an OLS fit used as the structural model (the OLS runs in
// the same graph). Left: residuals per tract. Right: the 3 x 3 coefficient covariance the
// operation wrote (intercept first) as bars, row by row. Errors that grow with x1 widen the
// variance of the slope; `HAC` adds the covariance of neighboring residuals through the kernel.

import {
  GPUInstrumentalVariablesCovariance,
  GPUOrdinaryLeastSquares,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights} from '../../inputs';
import {parameters} from './instrumental-variables-covariance.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawTractMap,
  MAP_SHARE,
  uploadRegressionMap
} from './shared/regression';
import {uploadWeightsBuffers} from './shared/weights';

const METHODS = ['HC0', 'HC1', 'HAC'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Residuals (left) and the IV covariance matrix, row by row (right).',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const design = builder.upload('design', 'float32', inputs.predictors);
    const instruments = builder.upload(
      'instruments',
      'float32',
      Float32Array.from({length: rowCount * 3}, (_, slot) => {
        const tract = Math.floor(slot / 3);
        const column = slot % 3;
        return column < 2
          ? inputs.predictors[tract * 2 + column]
          : inputs.predictors[tract * 2] * inputs.predictors[tract * 2 + 1];
      })
    );
    const response = builder.allocate('response', 'float32', rowCount);
    const residuals = builder.allocate('residuals', 'float32', rowCount);
    builder.graph.add(
      new GPUOrdinaryLeastSquares({
        predictors: design.view,
        response: response.view,
        predictorCount: 2,
        output: {
          status: {code: builder.allocate('ols-status', 'uint32', 1).view},
          coefficients: builder.allocate('ols-coefficients', 'float32', 3).view,
          standardErrors: builder.allocate('ols-standard-errors', 'float32', 3).view,
          tStatistics: builder.allocate('ols-t-statistics', 'float32', 3).view,
          summary: builder.allocate(
            'ols-summary',
            'float32',
            GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
          ).view,
          residuals: residuals.view
        }
      })
    );
    // HAC kernel: symmetric, unit diagonal, small weight on the neighbors (diagonally dominant).
    const neighbors = inputs.weights;
    const kernel = uploadWeightsBuffers(builder, 'kernel', createKernelCsr(neighbors, 1));
    const covariance = builder.allocate('covariance', 'float32', 9);
    const method = METHODS[values.method];
    builder.graph.add(
      new GPUInstrumentalVariablesCovariance({
        design: design.view,
        coefficientCount: 2,
        instruments: instruments.view,
        instrumentCount: 3,
        residuals: residuals.view,
        method,
        kernel: method === 'HAC' ? {...kernel.input, distances: undefined} : undefined,
        output: {covariance: covariance.view, status: builder.allocate('status', 'uint32', 1).view}
      })
    );
    return {
      outputs: {mesh: uploadRegressionMap(builder, inputs), residuals, covariance},
      update: current => {
        const errors = Float32Array.from(
          inputs.normals,
          (value, tract) =>
            value * 0.4 * (1 + current.heteroskedasticity * inputs.predictors[tract * 2])
        );
        const signal = createResponse(inputs, {noise: 0});
        response.write(signal.map((value, tract) => value + errors[tract]));
        kernel.weights.write(createKernelCsr(neighbors, current.kernelWeight).weights);
      }
    };
  },
  rebuildOn: ['method'],
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.mesh, outputs.residuals),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.covariance,
      ramp: 'diverging',
      rectangle: [MAP_SHARE + 0.08, 0.14, 0.97, 0.86]
    }
  ]
});

/** The neighbors CSR with a unit self slot in every row and `weight` on the neighbor slots. */
function createKernelCsr(neighbors: ReturnType<typeof createKnnWeights>, weight: number) {
  const offsets = new Uint32Array(neighbors.rowCount + 1);
  const columns: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < neighbors.rowCount; row++) {
    const rowNeighbors = Array.from(
      neighbors.neighbors.slice(neighbors.offsets[row], neighbors.offsets[row + 1])
    );
    for (const column of [...rowNeighbors, row].sort((first, second) => first - second)) {
      columns.push(column);
      weights.push(column === row ? 1 : weight);
    }
    offsets[row + 1] = columns.length;
  }
  return {
    rowCount: neighbors.rowCount,
    offsets,
    neighbors: Uint32Array.from(columns),
    weights: Float32Array.from(weights),
    distances: new Float32Array(columns.length)
  };
}
