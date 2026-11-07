// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-dataframe.variable-standardization',
  element: 'GPUVariableStandardization',
  title: 'Sample z-scores of Chicago tract poverty percentages',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'f-chicago-tracts',
    source: 'U.S. Census ACS and CDC/ATSDR SVI; Chicago showcase tracts',
    licence:
      'Census and CDC public domain; included nature-count data: iNaturalist contributors CC0 / CC BY 4.0 / CC BY-NC 4.0, non-commercial'
  },
  load: async () => import('./variable-standardization.run')
});
