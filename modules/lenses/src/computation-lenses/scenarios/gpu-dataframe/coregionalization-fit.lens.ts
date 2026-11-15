// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-dataframe.coregionalization-fit',
  element: 'GPUCoregionalizationFit',
  title: 'Coregionalization fit: a PSD matrix with negative cross semivariance',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'synthetic-spherical-coregionalization-bins-v1',
    source:
      'Repository-authored lag matrices from B = [[2, -1], [-1, 3]] and a unit-sill spherical structure of range 2; a fourth bin has zero pairs and is excluded.',
    licence: 'MIT'
  },
  load: async () => import('./coregionalization-fit.run')
});
