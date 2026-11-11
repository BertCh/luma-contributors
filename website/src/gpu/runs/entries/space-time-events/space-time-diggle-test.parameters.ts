// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Per-frame `parameters` of GPUSpaceTimeDiggleTest (largest distance and time lag of the D grid).
// The data control exchanges a share of the origin times between events.

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 2,
    max: 16,
    step: 1,
    default: 8,
    unit: 'km'
  },
  {id: 'maximumTimeLag', label: 'maximumTimeLag', min: 2, max: 40, step: 2, default: 16, unit: 'd'},
  {id: 'observedShare', label: 'Data: observed times', min: 0, max: 1, step: 0.05, default: 1}
];
