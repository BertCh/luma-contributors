// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'meuse-validation',
  title: 'Meuse topsoil metals and 40 m prediction grid',
  description:
    '155 measured topsoil samples with zinc, copper, lead, cadmium, binary lime, soil class, elevation, organic matter and river distance, plus the original 3,103 40 m-spaced query locations and Meuse river outline. Samples and predictions retain the source EPSG:28992 coordinates for planar analysis.',
  license: 'GPL-2.0-or-later (R package sp data)',
  attribution:
    'Meuse topsoil and river data distributed with R package sp 2.2-4; data collectors Ruud van Rijn and Mathieu Rikken; compiled by Edzer Pebesma.',
  sourceUrl: 'https://edzer.r-universe.dev/sp/data/meuse/json',
  approxBytes: 180_000,
  bbox: [5.721109, 50.95577, 5.76517, 50.992717]
} satisfies DatasetInfo;
