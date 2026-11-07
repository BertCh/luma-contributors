// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUProbitRegression of a binary outcome on two predictors. Left: the fitted probability of every
// tract. Right: the estimated coefficients (bars) against the latent-scale truth (dots), and the
// marginal effects below. A large latent noise flattens the probabilities and shrinks the slopes.

import {GPUProbitRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './probit-regression.parameters';
import {
  COEFFICIENT_RANGE,
  createCoefficientPanel,
  createRegressionInputs,
  drawCoefficientPanel,
  drawTractMap,
  MAP_SHARE,
  uploadRegressionMap
} from './shared/regression';

const LATENT_COEFFICIENTS = [-1.2, 2.4, -1.8] as const;

export default defineOperationRun({
  parameters,
  caption: 'Fitted probabilities (left) and probit coefficients against the truth (right).',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const coefficients = builder.allocate('coefficients', 'float32', 3);
    const fitted = builder.allocate('fitted', 'float32', rowCount);
    const marginalEffects = builder.allocate('marginal-effects', 'float32', 2);
    builder.graph.add(
      new GPUProbitRegression({
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        maximumIterations: values.maximumIterations,
        output: {
          coefficients: coefficients.view,
          standardErrors: builder.allocate('standard-errors', 'float32', 3).view,
          fitted: fitted.view,
          marginalEffects: marginalEffects.view,
          summary: builder.allocate('summary', 'float32', 4).view,
          status: builder.allocate('status', 'uint32', 1).view
        }
      })
    );
    const panel = createCoefficientPanel(builder, coefficients, 1, LATENT_COEFFICIENTS);
    return {
      outputs: {mesh: uploadRegressionMap(builder, inputs), fitted, marginalEffects, panel},
      update: current => {
        response.write(
          Float32Array.from({length: rowCount}, (_, tract) => {
            const latent =
              LATENT_COEFFICIENTS[0] +
              LATENT_COEFFICIENTS[1] * inputs.predictors[tract * 2] +
              LATENT_COEFFICIENTS[2] * inputs.predictors[tract * 2 + 1] +
              inputs.normals[tract] * current.noise;
            return latent > 0 ? 1 : 0;
          })
        );
        panel.truth.write(Float32Array.from(LATENT_COEFFICIENTS, value => value / current.noise));
      }
    };
  },
  rebuildOn: ['maximumIterations'],
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.mesh, outputs.fitted, {ramp: 'sequential', range: [0, 1]}),
    ...drawCoefficientPanel(outputs.panel).map(layer =>
      layer.layer === 'chart'
        ? {...layer, rectangle: [MAP_SHARE + 0.08, 0.42, 0.97, 0.9] as const}
        : layer
    ),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.marginalEffects,
      ramp: 'diverging',
      range: [-COEFFICIENT_RANGE[1] / 4, COEFFICIENT_RANGE[1] / 4],
      rectangle: [MAP_SHARE + 0.08, 0.08, 0.97, 0.34]
    }
  ]
});
