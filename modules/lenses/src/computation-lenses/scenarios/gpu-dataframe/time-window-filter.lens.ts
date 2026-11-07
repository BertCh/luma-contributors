// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-dataframe.time-window-filter',
  element: 'GPUTimeWindowFilter',
  title: 'Chicago nature observations in a summer window',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'f-chicago-nature',
    source: 'iNaturalist observations, Chicago, 2023; showcase chicago-nature v1',
    licence:
      'iNaturalist contributors; per-observation CC0 / CC BY 4.0 / CC BY-NC 4.0; non-commercial'
  },
  load: async () => import('./time-window-filter.run')
});
