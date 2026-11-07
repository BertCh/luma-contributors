// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-dataframe.class-breaks',
  element: 'GPUClassBreaks',
  title: 'Chicago tract poverty in five equal-interval classes',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'f-chicago-tracts',
    source: 'U.S. Census ACS and CDC/ATSDR SVI; Chicago showcase tracts',
    licence:
      'Census and CDC public domain; included nature-count data: iNaturalist contributors CC0 / CC BY 4.0 / CC BY-NC 4.0, non-commercial'
  },
  load: async () => import('./class-breaks.run')
});
