// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialPanelLagMaximumLikelihood on six periods of tract data with entity effects. Left: the
// residuals of the first period. Right: the coefficients the operation wrote (intercept, two
// slopes, rho, and the random-effects share phi) as bars, with the true values as dots.

import {GPUSpatialPanelLagMaximumLikelihood} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {parameters} from './spatial-panel-lag-ml.parameters';
import {TRUE_COEFFICIENTS} from './shared/regression';
import {definePanelRun} from './shared/panel';

export default definePanelRun({
  parameters,
  caption: 'Panel lag model by maximum likelihood: estimates (bars) against the truth (dots).',
  process: 'lag',
  rebuildOn: ['effects', 'maximumIterations'],
  addOperation: ({builder, values, weights, predictors, response, periodCount, residuals}) => {
    const rowCount = response.length;
    const coefficients = builder.allocate('coefficients', 'float32', 5);
    builder.graph.add(
      new GPUSpatialPanelLagMaximumLikelihood({
        weights,
        periodCount,
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        effects: values.effects === 0 ? 'random' : 'unit',
        maximumIterations: values.maximumIterations,
        output: {
          coefficients: coefficients.view,
          covariance: builder.allocate('covariance', 'float32', 16).view,
          summary: builder.allocate('summary', 'float32', 6).view,
          status: builder.allocate('status', 'uint32', 1).view,
          fitted: builder.allocate('fitted', 'float32', rowCount).view,
          residuals: residuals.view
        }
      })
    );
    return {
      source: coefficients,
      columnCount: 1,
      truthOf: strength => [...TRUE_COEFFICIENTS, strength, 0.5]
    };
  }
});
