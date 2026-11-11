// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Per-frame `parameters` of GPUSpaceTimeKFunction: the largest distance and time lag of the grid.

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 40,
    max: 400,
    step: 20,
    default: 200,
    unit: 'km'
  },
  {
    id: 'maximumTimeLag',
    label: 'maximumTimeLag',
    min: 0.5,
    max: 8,
    step: 0.25,
    default: 3,
    unit: 'h'
  }
];
