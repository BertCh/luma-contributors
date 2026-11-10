// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame `window` of GPUTimeWindowFilter: start, end and the start fade duration.
export const parameters: GpuSceneParameter[] = [
  {id: 'start', label: 'window: start', min: 0, max: 17, step: 0.5, default: 6, unit: 'h'},
  {id: 'end', label: 'window: end', min: 4, max: 18, step: 0.5, default: 14, unit: 'h'},
  {
    id: 'startFadeDuration',
    label: 'window: startFadeDuration',
    min: 0,
    max: 8,
    step: 0.5,
    default: 3,
    unit: 'h'
  }
];
