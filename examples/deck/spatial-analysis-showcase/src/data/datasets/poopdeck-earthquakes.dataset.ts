// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-earthquakes',
  title: 'Global ComCat events M4+, 2020 to December 30, 2024',
  description:
    'Archived selection of 77,231 USGS ComCat magnitude 4+ events from January 1, 2020 through December 30, 2024, via poopdeck.gl, sorted by time. Includes 77,215 earthquakes and 16 other seismic event types. This fixed snapshot is not a claim of current catalog completeness; the upstream archive description has an inconsistent larger count.',
  license: 'Public domain (US Government work)',
  attribution: 'USGS Earthquake Catalog (ComCat), via poopdeck.gl',
  sourceUrl: 'https://earthquake.usgs.gov/fdsnws/event/1/',
  approxBytes: 1_700_000,
  bbox: [-180, -83, 180, 87]
} satisfies DatasetInfo;
