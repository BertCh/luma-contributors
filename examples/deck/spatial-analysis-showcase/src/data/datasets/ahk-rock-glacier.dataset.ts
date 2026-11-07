// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'ahk-rock-glacier',
  title: 'Äußeres Hochebenkar rock glacier, 2017 and 2018A lidar',
  description:
    'Two genuine terrestrial-laser-scan epochs from a 37 m rock-glacier patch: 5,192 and 5,407 original returns retained by 0.6 m voxel sampling, plus 258 provided core points and normals. A companion 1 m grid has 718 cells with at least three returns in both epochs. The supplied registration transform is applied to 2018A; local refitting is not part of preparation.',
  license: 'CC BY 4.0',
  attribution:
    'Winiwarter, Anders, Zahs, Hämmerle and Höfle (2021), heiDATA, doi:10.11588/data/XHYB10; sample via 3DGeo Heidelberg py4dgeo-test-data (CC BY 4.0)',
  sourceUrl: 'https://doi.org/10.11588/data/XHYB10',
  approxBytes: 400_000,
  bbox: [11.00517852176768, 46.83819343604209, 11.005675070657384, 46.83852989854902]
} satisfies DatasetInfo;
