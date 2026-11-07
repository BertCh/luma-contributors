// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// tiles is the number of horizontal tiles the DEM arrives in (the operation's `tiles` list, compile-time);
// streamThreshold is the stream mask each tile carries.
export const parameters: GpuSceneParameter[] = [
  {id: 'tiles', label: 'tiles', min: 1, max: 4, step: 1, default: 3},
  {
    id: 'streamThreshold',
    label: 'streamThreshold',
    min: 5,
    max: 200,
    step: 5,
    default: 60,
    unit: ' cells'
  }
];
