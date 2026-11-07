// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-dataframe.variogram-model-fit',
  element: 'GPUVariogramModelFit',
  title: 'Resident variogram fitting: three models on synthetic empirical bins',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'synthetic-exponential-variogram-bins-v1',
    source:
      'Twenty repository-authored positive lags from an exponential model with nugget 0.2, partial sill 1.5 and effective range 2.2.',
    licence: 'MIT'
  },
  load: async () => import('./variogram-model-fit.run')
});
