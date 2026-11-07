// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-dataframe.spatial-correlogram',
  element: 'GPUSpatialCorrelogram',
  title: "Moran's I of Hurricane Helene rainfall by distance band",
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'f-ghcn-stations',
    source: 'NOAA NCEI GHCN-Daily via AWS Open Data (noaa-ghcn-pds)',
    licence: 'NOAA NCEI GHCN-Daily, public domain'
  },
  load: async () => import('./spatial-correlogram.run')
});
