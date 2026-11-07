// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-dataframe.point-pattern-indices',
  element: 'GPUPointPatternIndices',
  title: 'Clark-Evans index and quadrat test for Chicago iNaturalist observations',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'f-chicago-nature',
    source: 'iNaturalist observations, Chicago 2023 (api.inaturalist.org)',
    licence:
      'iNaturalist contributors; per-observation CC0 / CC BY 4.0 / CC BY-NC 4.0; non-commercial'
  },
  load: async () => import('./point-pattern-indices.run')
});
