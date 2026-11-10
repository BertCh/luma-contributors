// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialErrorMaximumLikelihood on a response with spatially correlated errors. Left: the
// filtered residuals per tract. Right: the estimated coefficients (intercept, slopes and the error
// parameter lambda last) as bars, with the true values as dots.

import {
  GPUOrdinaryLeastSquares,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUSpatialErrorMaximumLikelihood} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {drawTractMap, RESIDUAL_RANGE} from './shared/regression';
import {parameters} from './spatial-error-maximum-likelihood.parameters';
import {defineTableModelRun} from './shared/table-model';
import type {RunBuffer} from '../../define-run';

export default defineTableModelRun<{olsResiduals: RunBuffer<'float32'>}>({
  parameters,
  caption: 'Maximum-likelihood error model: estimates (bars) against the truth (dots).',
  process: 'error',
  rebuildOn: ['maximumPower'],
  addOperation: ({
    builder,
    values,
    weights,
    predictors,
    response,
    table,
    summary,
    status,
    residuals
  }) => {
    // Plain OLS residuals of the same response, to swipe against the filtered ones.
    const olsResiduals = builder.allocate('ols-residuals', 'float32', response.length);
    builder.graph.add(
      new GPUOrdinaryLeastSquares({
        predictors: predictors.view,
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
          residuals: olsResiduals.view,
          fitted: builder.allocate('ols-fitted', 'float32', response.length).view
        }
      })
    );
    builder.graph.add(
      new GPUSpatialErrorMaximumLikelihood({
        weights,
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        maximumPower: values.maximumPower,
        output: {
          status: {code: status.view},
          table: table.view,
          summary: summary.view,
          residuals: residuals.view
        }
      })
    );
    return {olsResiduals};
  },
  // Left of the curtain the map shows plain OLS residuals, right of it the residuals filtered by the error model.
  interaction: {
    kind: 'swipe',
    hint: 'swipe OLS and filtered',
    rest: state => state.toUnit([state.inputs.mapWidth * 0.5, 0.5]),
    resolve(pointer, {inputs, values}) {
      const curtain = Math.min(Math.max(pointer.x, 0), inputs.mapWidth);
      return {
        curtain,
        glyphs: [{shape: 'curtain', x: curtain}],
        readout: `OLS | filtered residuals · lambda ${values.strength}`
      };
    }
  },
  drawMap: ({inputs, outputs, interaction}) => {
    const curtain = interaction.curtain ?? inputs.mapWidth * 0.5;
    const clipped = (residuals: typeof outputs.residuals, side: 'left' | 'right') =>
      drawTractMap(inputs, outputs.mesh, residuals, {ramp: 'diverging', range: RESIDUAL_RANGE}).map(
        layer => (layer.layer === 'polygons' ? {...layer, clip: {side, x: curtain}} : layer)
      );
    return [...clipped(outputs.olsResiduals, 'left'), ...clipped(outputs.residuals, 'right')];
  }
});
