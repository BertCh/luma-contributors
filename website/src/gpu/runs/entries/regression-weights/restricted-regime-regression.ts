// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURestrictedRegimeRegression fits two regimes (west and east) with chosen coefficients shared.
// Left: fitted values per tract. Right: the restricted model's coefficients (a shared coefficient is
// one bar, a regime-specific one is two) and below them the Chow F statistic against the fully
// varying model, which grows when a shared coefficient is truly different between regimes.

import {GPURestrictedRegimeRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './restricted-regime-regression.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawTractMap,
  MAP_SHARE,
  TRUE_COEFFICIENTS,
  uploadRegressionMap
} from './shared/regression';
import {addTableColumn} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Restricted regimes: coefficients (right), Chow F below, fitted map (left).',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const common = [
      values.commonIntercept === 1,
      values.commonFirst === 1,
      values.commonSecond === 1
    ];
    const coefficientCount = common.reduce((total, isCommon) => total + (isCommon ? 1 : 2), 0);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const regimes = builder.upload('regimes', 'uint32', inputs.regimes);
    const coefficients = builder.allocate('coefficients', 'float32', coefficientCount);
    const fitted = builder.allocate('fitted', 'float32', rowCount);
    const chow = builder.allocate('chow', 'float32', 3);
    builder.graph.add(
      new GPURestrictedRegimeRegression({
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        regimes: regimes.view,
        regimeCount: 2,
        common,
        output: {
          coefficients: coefficients.view,
          covariance: builder.allocate('covariance', 'float32', coefficientCount ** 2).view,
          fitted: fitted.view,
          chow: chow.view,
          status: builder.allocate('status', 'uint32', 1).view
        }
      })
    );
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        fitted,
        coefficients,
        chowStatistic: addTableColumn(builder, 'chow', chow, 3, 0, 1)
      },
      update: current =>
        response.write(
          createResponse(inputs, {
            noise: 0.4,
            coefficients: tract =>
              inputs.regimes[tract] === 0
                ? TRUE_COEFFICIENTS
                : [
                    TRUE_COEFFICIENTS[0] + 0.6 * current.contrast,
                    TRUE_COEFFICIENTS[1] - 1.1 * current.contrast,
                    TRUE_COEFFICIENTS[2] + 0.9 * current.contrast
                  ]
          })
        )
    };
  },
  rebuildOn: ['commonIntercept', 'commonFirst', 'commonSecond'],
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.mesh, outputs.fitted, {ramp: 'sequential'}),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.coefficients,
      ramp: 'diverging',
      range: [-3, 3],
      rectangle: [MAP_SHARE + 0.08, 0.42, 0.97, 0.9]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.chowStatistic,
      color: [0.95, 0.6, 0.3, 0.95],
      range: [0, 60],
      rectangle: [MAP_SHARE + 0.08, 0.08, 0.97, 0.34]
    }
  ]
});
