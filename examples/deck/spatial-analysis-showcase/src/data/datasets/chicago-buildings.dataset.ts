// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-buildings',
  title: 'Chicago building footprints (Loop and Near West Side)',
  description:
    '1,706 active City of Chicago building footprints in a bounded Loop and Near West Side window, with source building IDs and incomplete story-count fields.',
  license: 'City of Chicago Data Portal Terms of Use',
  attribution: 'City of Chicago Data Portal — Building Footprints',
  sourceUrl: 'https://data.cityofchicago.org/Buildings/Building-Footprints-current-/hz9b-7nh8',
  approxBytes: 2_300_000,
  bbox: [-87.65, 41.87, -87.62, 41.895]
} satisfies DatasetInfo;
