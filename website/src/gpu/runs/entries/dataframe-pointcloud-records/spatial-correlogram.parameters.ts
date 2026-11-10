// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialCorrelogram's compile-time `bandCount` and `bandMode`, its per-frame
// `maximumDistance`, and the noise of the observed field.
export const parameters: GpuSceneParameter[] = [
  {id: 'bandCount', label: 'bandCount', min: 4, max: 24, step: 2, default: 12},
  {
    id: 'bandMode',
    label: 'bandMode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['cumulative', 'annulus']
  },
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 0.1,
    max: 0.6,
    step: 0.05,
    default: 0.4
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 3, step: 0.1, default: 0}
];
