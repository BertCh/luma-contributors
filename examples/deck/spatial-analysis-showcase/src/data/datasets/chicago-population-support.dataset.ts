// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-population-support',
  title: 'Hyde Park population and WorldCover support',
  description:
    'All 12 whole 2020 Census tracts assigned to Chicago community area 41, with SVI 2022 population and race counts and an equal-area 2021 ESA WorldCover crop on a 500 m analysis grid.',
  license: 'Public domain (US Census Bureau, CDC/ATSDR); ESA WorldCover CC BY 4.0',
  attribution:
    'US Census Bureau 2022 TIGER/Cartographic Boundary; CDC/ATSDR SVI 2022; © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data 2021 processed by WorldCover consortium',
  sourceUrl: 'https://esa-worldcover.org/en/data-access',
  approxBytes: 350_000,
  bbox: [-87.606407, 41.785874, -87.575801, 41.802512]
} satisfies DatasetInfo;
