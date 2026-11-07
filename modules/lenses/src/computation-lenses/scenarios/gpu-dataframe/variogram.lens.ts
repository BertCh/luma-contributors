// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-dataframe.variogram',
  element: 'GPUVariogram',
  title: 'Empirical semivariogram of Hurricane Helene rainfall at 1,200 stations',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'f-ghcn-stations',
    source: 'NOAA NCEI GHCN-Daily via AWS Open Data (noaa-ghcn-pds)',
    licence: 'NOAA NCEI GHCN-Daily, public domain'
  },
  load: async () => import('./variogram.run')
});
