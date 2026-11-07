// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialTwoStageLeastSquares on a spatial-lag response. Left: structural residuals per tract.
// Right: the coefficient table's estimates (intercept, two slopes and the spatial lag rho) as bars,
// with the true values as dots.

import {GPUSpatialTwoStageLeastSquares} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {parameters} from './spatial-two-stage-least-squares.parameters';
import {defineTableModelRun} from './shared/table-model';

export default defineTableModelRun({
  parameters,
  caption: 'Spatial 2SLS estimates (bars) against the true coefficients (dots).',
  process: 'lag',
  rebuildOn: ['instrumentOrder'],
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
      new GPUSpatialTwoStageLeastSquares({
        weights,
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        instrumentOrder: values.instrumentOrder === 1 ? 1 : 2,
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
