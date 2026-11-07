// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialRegimeRegression fits a separate intercept and slopes in every regime (vertical bands
// of tracts). Left: the fitted value of each tract, which jumps at regime boundaries. Right: the
// coefficients regime by regime (bars) with the true values (dots).

import {GPUSpatialRegimeRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './spatial-regime-regression.parameters';
import {
  createCoefficientPanel,
  createRegressionInputs,
  createResponse,
  drawCoefficientPanel,
  drawTractMap,
  TRUE_COEFFICIENTS,
  uploadRegressionMap
} from './shared/regression';

/** True coefficients of regime `regime`, drifting away from the base set by `contrast`. */
function getRegimeCoefficients(regime: number, contrast: number): [number, number, number] {
  return [
    TRUE_COEFFICIENTS[0] + 0.6 * regime * contrast,
    TRUE_COEFFICIENTS[1] - 1.1 * regime * contrast,
    TRUE_COEFFICIENTS[2] + 0.9 * regime * contrast
  ];
}

export default defineOperationRun({
  parameters,
  caption: 'Fitted values per regime and the per-regime coefficients against the truth.',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const regimeCount = values.regimeCount;
    const regimeOf = (tract: number) =>
      Math.min(
        regimeCount - 1,
        Math.floor((inputs.tracts.centers[tract * 2] / inputs.mapWidth) * regimeCount)
      );
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const regimes = builder.upload(
      'regimes',
      'uint32',
      Uint32Array.from({length: rowCount}, (_, tract) => regimeOf(tract))
    );
    const coefficients = builder.allocate('coefficients', 'float32', regimeCount * 3);
    const fitted = builder.allocate('fitted', 'float32', rowCount);
    builder.graph.add(
      new GPUSpatialRegimeRegression({
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        regimes: regimes.view,
        regimeCount,
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          status: builder.allocate('status', 'uint32', regimeCount).view,
          counts: builder.allocate('counts', 'uint32', regimeCount).view
        }
      })
    );
    const panel = createCoefficientPanel(
      builder,
      coefficients,
      1,
      Array.from({length: regimeCount * 3}, () => 0)
    );
    return {
      outputs: {mesh: uploadRegressionMap(builder, inputs), fitted, panel},
      update: current => {
        response.write(
          createResponse(inputs, {
            noise: current.noise,
            coefficients: tract => getRegimeCoefficients(regimeOf(tract), current.contrast)
          })
        );
        panel.truth.write(
          Float32Array.from(
            {length: regimeCount * 3},
            (_, slot) => getRegimeCoefficients(Math.floor(slot / 3), current.contrast)[slot % 3]
          )
        );
      }
    };
  },
  rebuildOn: ['regimeCount'],
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.mesh, outputs.fitted, {ramp: 'sequential'}),
    ...drawCoefficientPanel(outputs.panel)
  ]
});
