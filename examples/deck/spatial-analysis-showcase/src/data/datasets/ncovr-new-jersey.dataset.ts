// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'ncovr-new-jersey',
  title: 'NCOVR historical New Jersey county units, 1970–1990',
  description:
    'All 21 New Jersey units in GeoDa’s historical common-support county geography, with observed unemployment, family poverty, log population density, decennial population counts and source-derived queen/rook contiguity.',
  license: 'Source data provided as-is with no warranties; no separate license asserted',
  attribution:
    'Messner, Anselin, Hawkins, Deane, Tolnay and Baller (2000), NCOVR; GeoDa archive prepared by Luc Anselin',
  sourceUrl: 'https://geodacenter.github.io/data-and-lab/ncovr/',
  approxBytes: 120000,
  bbox: [-75.570587, 38.95657, -73.896568, 41.350479]
} satisfies DatasetInfo;
