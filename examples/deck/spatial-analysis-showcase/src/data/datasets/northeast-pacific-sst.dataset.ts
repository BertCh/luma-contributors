// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'northeast-pacific-sst',
  title: 'Northeast Pacific monthly temperatures, 2012–2017 (NOAA ERSSTv5)',
  description:
    '72 real monthly reconstructed sea-surface temperature grids and NOAA climatological anomalies. A 26 × 21 subset at the original 2° spacing preserves missing land cells and calendar-month sampling.',
  license: 'NOAA: no constraints on data access or use; free redistribution',
  attribution: 'NOAA National Centers for Environmental Information; NOAA CoastWatch ERDDAP',
  sourceUrl:
    'https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00927',
  approxBytes: 330000,
  bbox: [-161, 19, -109, 61]
} satisfies DatasetInfo;
