// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointDensity's `gridSize`, `statistic` and `smoothing` props.
export const parameters: GpuSceneParameter[] = [
  {id: 'gridSize', label: 'gridSize', min: 8, max: 48, step: 4, default: 24, unit: ' rows'},
  {
    id: 'statistic',
    label: 'statistic',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['count', 'sum', 'mean']
  },
  {id: 'smoothing', label: 'smoothing radius', min: 0, max: 3, step: 1, default: 0, unit: ' cells'}
];
