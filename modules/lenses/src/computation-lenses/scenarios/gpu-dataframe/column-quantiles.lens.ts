// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-dataframe.column-quantiles',
  element: 'GPUColumnQuantiles',
  inputBindings: [{port: 'values', role: 'poverty150Pct', required: true, missing: 'exclude'}],
  title: 'Chicago tract poverty quartiles and a 5-95 percentile filter',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'f-chicago-tracts',
    source: 'U.S. Census ACS and CDC/ATSDR SVI; Chicago showcase tracts',
    licence:
      'Census and CDC public domain; included nature-count data: iNaturalist contributors CC0 / CC BY 4.0 / CC BY-NC 4.0, non-commercial'
  },
  load: async () => import('./column-quantiles.run')
});
