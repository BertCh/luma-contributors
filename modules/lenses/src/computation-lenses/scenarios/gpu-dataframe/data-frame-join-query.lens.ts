// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, DEFAULT_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-dataframe.data-frame-join-query',
  element: 'GPUDataFrameJoinQuery',
  title: 'Chicago wildlife observations joined to their census tracts',
  rungs: DEFAULT_LENS_RUNGS,
  dataset: {
    id: 'f-chicago-nature',
    source:
      'iNaturalist observations in Chicago, 2023, and census tracts (showcase chicago-nature and chicago-tracts v1)',
    licence:
      'iNaturalist contributors; per-observation CC0 / CC BY 4.0 / CC BY-NC 4.0; non-commercial. Census and CDC data: public domain'
  },
  load: async () => import('./data-frame-join-query.run')
});
