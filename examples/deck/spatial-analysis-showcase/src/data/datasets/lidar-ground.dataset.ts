// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'lidar-ground',
  title: 'USGS 3DEP King County lidar (180 m sample)',
  description:
    'A bounded 180 m square from USGS 3DEP King County 2021 tile 2282. The source has 6,373,072 returns; the sample keeps one original return per occupied 0.5 m cell (116,213 points), with LAS classification and NAVD88 elevation.',
  license: 'Public domain (U.S. Government work; USGS 3DEP)',
  attribution: 'U.S. Geological Survey, 3D Elevation Program (3DEP), King County 2021',
  sourceUrl: 'https://data.usgs.gov/datacatalog/data/USGS%3Ab7e353d2-325f-4fc6-8d95-01254705638a',
  approxBytes: 2_789_112,
  bbox: [-122.34098606255156, 47.595064533227394, -122.33873237750235, 47.59671148171778]
} satisfies DatasetInfo;
