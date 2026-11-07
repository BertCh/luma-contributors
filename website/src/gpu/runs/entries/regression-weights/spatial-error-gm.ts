// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialErrorGM estimates the spatial error parameter by generalized moments. Left: residuals
// per tract. Right: the estimated coefficients (intercept, slopes, lambda last) as bars, with the
// true values as dots.

import {GPUSpatialErrorGM} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {parameters} from './spatial-error-gm.parameters';
import {defineTableModelRun} from './shared/table-model';

export default defineTableModelRun({
  parameters,
  caption: 'Generalized-moments error model: estimates (bars) against the truth (dots).',
  process: 'error',
  rebuildOn: ['estimator'],
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
      new GPUSpatialErrorGM({
        weights,
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        estimator: values.estimator === 0 ? 'homoskedastic' : 'kelejian-prucha',
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
