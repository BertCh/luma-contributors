// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialLagMaximumLikelihood searches rho by maximum likelihood on a spatial-lag response.
// Left: residuals per tract. Right: the estimated coefficients (intercept, slopes and rho last) as
// bars, with the true values as dots.

import {GPUSpatialLagMaximumLikelihood} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {parameters} from './spatial-lag-maximum-likelihood.parameters';
import {defineTableModelRun} from './shared/table-model';

export default defineTableModelRun({
  parameters,
  caption: 'Maximum-likelihood lag model: estimates (bars) against the truth (dots).',
  process: 'lag',
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
      new GPUSpatialLagMaximumLikelihood({
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
