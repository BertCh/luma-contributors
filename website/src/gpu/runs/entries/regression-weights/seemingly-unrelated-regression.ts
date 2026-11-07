// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSeeminglyUnrelatedRegression fits two equations on the same tracts whose errors are
// correlated. Left: residuals of the first equation. Right: the six coefficients (bars) with the
// true values (dots); below, the 2 x 2 residual covariance the operation estimated (variances and
// the cross-equation covariance, which follows the correlation slider).

import {GPUSeeminglyUnrelatedRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './seemingly-unrelated-regression.parameters';
import {
  createCoefficientPanel,
  createRegressionInputs,
  drawCoefficientPanel,
  drawTractMap,
  MAP_SHARE,
  uploadRegressionMap
} from './shared/regression';

const COEFFICIENTS = [1, 2, -1.5, 0.5, -1, 2] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Two-equation SUR: coefficients (bars) against the truth (dots), residual covariance below.',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs) {
    const rowCount = inputs.tracts.count;
    const predictors = builder.upload(
      'predictors',
      'float32',
      Float32Array.from([...inputs.predictors, ...inputs.predictors])
    );
    const responses = builder.allocate('responses', 'float32', rowCount * 2);
    const coefficients = builder.allocate('coefficients', 'float32', 6);
    const residuals = builder.allocate('residuals', 'float32', rowCount * 2);
    const residualCovariance = builder.allocate('residual-covariance', 'float32', 4);
    builder.graph.add(
      new GPUSeeminglyUnrelatedRegression({
        equationCount: 2,
        predictorCount: 2,
        rowCount,
        predictors: predictors.view,
        responses: responses.view,
        output: {
          coefficients: coefficients.view,
          covariance: builder.allocate('covariance', 'float32', 36).view,
          residualCovariance: residualCovariance.view,
          residuals: residuals.view,
          status: builder.allocate('status', 'uint32', 1).view
        }
      })
    );
    const panel = createCoefficientPanel(builder, coefficients, 1, COEFFICIENTS);
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        panel,
        residualCovariance,
        residualBuffer: residuals
      },
      update: current => {
        const first = new Float32Array(rowCount);
        const second = new Float32Array(rowCount);
        const mix = Math.sqrt(1 - current.correlation ** 2);
        for (let tract = 0; tract < rowCount; tract++) {
          const own = inputs.normals[(tract * 7 + 3) % rowCount];
          const error = inputs.normals[tract];
          const signalFirst =
            COEFFICIENTS[0] +
            COEFFICIENTS[1] * inputs.predictors[tract * 2] +
            COEFFICIENTS[2] * inputs.predictors[tract * 2 + 1];
          const signalSecond =
            COEFFICIENTS[3] +
            COEFFICIENTS[4] * inputs.predictors[tract * 2] +
            COEFFICIENTS[5] * inputs.predictors[tract * 2 + 1];
          first[tract] = signalFirst + error * current.noise;
          second[tract] = signalSecond + (current.correlation * error + mix * own) * current.noise;
        }
        responses.write(Float32Array.from([...first, ...second]));
      }
    };
  },
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.mesh, outputs.residualBuffer),
    ...drawCoefficientPanel(outputs.panel).map(layer =>
      layer.layer === 'chart'
        ? {...layer, rectangle: [MAP_SHARE + 0.08, 0.42, 0.97, 0.9] as const}
        : layer
    ),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.residualCovariance,
      ramp: 'diverging',
      range: [-0.4, 0.4],
      rectangle: [MAP_SHARE + 0.08, 0.08, 0.97, 0.34]
    }
  ]
});
