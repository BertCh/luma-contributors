// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// One bandwidth per coefficient, as GPUMultiscaleGeographicallyWeightedRegression's `parameters`
// buffer holds them. The first slope truly varies, the second is constant.
export const parameters: GpuSceneParameter[] = [
  {id: 'interceptBandwidth', label: 'intercept bandwidth', min: 0.4, max: 3, step: 0.1, default: 3},
  {
    id: 'firstBandwidth',
    cardScrub: true,
    label: 'x1 bandwidth',
    min: 0.2,
    max: 3,
    step: 0.1,
    default: 0.5
  },
  {id: 'secondBandwidth', label: 'x2 bandwidth', min: 0.25, max: 3, step: 0.1, default: 3}
];
