// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeneralizedGeographicallyWeightedRegression fits a local Poisson or logistic model at every
// tract center. Left: the local slope of the first predictor, which drifts from west to east in
// the data. Right: the fitted mean (counts) or probability.

import {GPUGeneralizedGeographicallyWeightedRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './generalized-gwr.parameters';
import {
  createRegressionInputs,
  drawTractMap,
  DUAL_MAP_SHARE,
  uploadRegressionMap
} from './shared/regression';
import {addTableColumn} from './shared/weights';

/** Samples a Poisson count by inversion with a fixed uniform draw. */
function samplePoisson(mean: number, uniform: number): number {
  let probability = Math.exp(-mean);
  let cumulative = probability;
  let count = 0;
  while (uniform > cumulative && count < 60) {
    count++;
    probability *= mean / count;
    cumulative += probability;
  }
  return count;
}

export default defineOperationRun({
  parameters,
  caption: 'Local slope of the first predictor (left) and fitted mean or probability (right).',
  createInputs: ({random, world}) =>
    createRegressionInputs(random, world, 10, DUAL_MAP_SHARE, true),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const isPoisson = values.family === 0;
    const positions = builder.upload('positions', 'float32x2', inputs.tracts.centers);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const coefficients = builder.allocate('coefficients', 'float32', rowCount * 3);
    const fitted = builder.allocate('fitted', 'float32', rowCount);
    builder.graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        family: isPoisson ? 'poisson' : 'logistic',
        kernel: values.kernel === 0 ? 'gaussian' : 'bisquare',
        positions: positions.view,
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        parameters: parameterBuffer.view,
        maximumIterations: 30,
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          status: {
            codes: builder.allocate('codes', 'uint32', rowCount).view,
            iterationCount: builder.allocate('iteration-count', 'uint32', rowCount).view
          }
        }
      })
    );
    return {
      outputs: {
        meshes: [
          uploadRegressionMap(builder, inputs),
          uploadRegressionMap(builder, inputs, inputs.mapWidth * 1.14)
        ],
        slope: addTableColumn(builder, 'local-slope', coefficients, 3, 1, rowCount),
        fitted
      },
      update: current => {
        response.write(
          Float32Array.from({length: rowCount}, (_, tract) => {
            const east = inputs.tracts.centers[tract * 2] / inputs.mapWidth;
            const slope = 0.2 + current.drift * east;
            const linear =
              (isPoisson ? 0.3 : -0.8) +
              slope * inputs.predictors[tract * 2] * (isPoisson ? 1 : 2.5) -
              0.8 * inputs.predictors[tract * 2 + 1];
            return isPoisson
              ? samplePoisson(Math.exp(linear), inputs.uniforms[tract])
              : inputs.uniforms[tract] < 1 / (1 + Math.exp(-linear))
                ? 1
                : 0;
          })
        );
        parameterBuffer.write(Float32Array.of(current.bandwidth, 1e-4));
      }
    };
  },
  rebuildOn: ['family', 'kernel'],
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.meshes[0], outputs.slope, {ramp: 'sequential'}),
    ...drawTractMap(inputs, outputs.meshes[1], outputs.fitted, {ramp: 'sequential'})
  ]
});
