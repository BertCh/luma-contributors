// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialPanelErrorMaximumLikelihood: unit fixed-effects spatial error model over six periods.
// Left: residuals of the first period. Right: the coefficient table (the unit intercepts are
// absorbed, so the first bar is empty; two slopes; lambda last) as bars with the true values as dots.

import {GPUSpatialPanelErrorMaximumLikelihood} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {parameters} from './spatial-panel-error-ml.parameters';
import {TRUE_COEFFICIENTS} from './shared/regression';
import {definePanelRun} from './shared/panel';

export default definePanelRun({
  parameters,
  caption: 'Panel error model by maximum likelihood: estimates (bars) against the truth (dots).',
  process: 'error',
  rebuildOn: ['maximumPower', 'traceMode'],
  addOperation: ({builder, values, weights, predictors, response, periodCount, residuals}) => {
    const table = builder.allocate('table', 'float32', 16);
    builder.graph.add(
      new GPUSpatialPanelErrorMaximumLikelihood({
        weights,
        periodCount,
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        maximumPower: values.maximumPower,
        traceMode: values.traceMode === 0 ? 'exact' : 'stochastic',
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
