// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'dixie-spectral',
  title: 'Dixie Fire six-band spectral evidence (Sentinel-2, 60 m)',
  description:
    'Pinned July 13 and September 21, 2021 scenes in a 15 km UTM window. Six calibrated surface-reflectance bands, strict native-pixel scene-classification masks, and separate land-only support preserve the evidence behind spectral and topology comparisons.',
  license: 'Copernicus Sentinel data: free, full and open',
  attribution: 'Contains modified Copernicus Sentinel data 2021',
  sourceUrl: 'https://registry.opendata.aws/sentinel-2-l2a-cogs/',
  approxBytes: 3310000,
  bbox: [-121.08993894007818, 40.103921542233465, -120.90991598607712, 40.235939588107]
} satisfies DatasetInfo;
