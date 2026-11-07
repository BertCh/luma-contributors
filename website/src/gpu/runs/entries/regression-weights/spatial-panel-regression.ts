// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialPanelRegression: a spatial-lag 2SLS over six periods of the same tracts. Left: the
// residuals of the first period. Right: the coefficient table (intercept, slopes, rho) as bars with
// the true values as dots. With `effects: 'unit'` the entity intercepts are absorbed (the intercept
// bar disappears) and the slopes stay accurate however large the entity effects are.

import {GPUSpatialPanelRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {parameters} from './spatial-panel-regression.parameters';
import {TRUE_COEFFICIENTS} from './shared/regression';
import {definePanelRun} from './shared/panel';

export default definePanelRun({
  parameters,
  caption: 'Panel 2SLS: estimates (bars) against the true coefficients (dots).',
  process: 'lag',
  rebuildOn: ['effects'],
  addOperation: ({builder, values, weights, predictors, response, periodCount, residuals}) => {
    const table = builder.allocate('table', 'float32', 16);
    builder.graph.add(
      new GPUSpatialPanelRegression({
        weights,
        periodCount,
        effects: values.effects === 0 ? 'pooled' : 'unit',
        predictorCount: 2,
        predictors: predictors.view,
        response: response.view,
        output: {
          status: {code: builder.allocate('status', 'uint32', 1).view},
          table: table.view,
          summary: builder.allocate('summary', 'float32', 32).view,
          residuals: residuals.view
        }
      })
    );
    return {source: table, columnCount: 4, truthOf: strength => [...TRUE_COEFFICIENTS, strength]};
  }
});
