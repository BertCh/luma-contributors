// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialErrorMaximumLikelihood on a response with spatially correlated errors. Left: the
// filtered residuals per tract. Right: the estimated coefficients (intercept, slopes and the error
// parameter lambda last) as bars, with the true values as dots.

import {GPUSpatialErrorMaximumLikelihood} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {parameters} from './spatial-error-maximum-likelihood.parameters';
import {defineTableModelRun} from './shared/table-model';

export default defineTableModelRun({
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
  }
});
