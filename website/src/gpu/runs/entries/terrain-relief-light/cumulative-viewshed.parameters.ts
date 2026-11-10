// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainCumulativeViewshed({observers}) and getGPUTerrainLineOfSightParameterValues({observerHeight, maximumDistance}).
export const parameters: GpuSceneParameter[] = [
  {id: 'observerCount', label: 'observers: count', min: 1, max: 8, step: 1, default: 5},
  {
    id: 'observerHeight',
    label: 'observerHeight',
    min: 0,
    max: 120,
    step: 2,
    default: 20,
    unit: ' m'
  },
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 300,
    max: 5000,
    step: 100,
    default: 3000,
    unit: ' m'
  }
];
