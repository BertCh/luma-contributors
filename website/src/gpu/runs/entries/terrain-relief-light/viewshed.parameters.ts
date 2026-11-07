// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUTerrainViewshedParameterValues({observer, observerHeight, maximumDistance}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'observerColumn',
    label: 'observer: column',
    min: 0,
    max: 100,
    step: 1,
    default: 30,
    unit: '%'
  },
  {id: 'observerRow', label: 'observer: row', min: 0, max: 100, step: 1, default: 55, unit: '%'},
  {
    id: 'observerHeight',
    label: 'observerHeight',
    min: 0,
    max: 120,
    step: 2,
    default: 10,
    unit: ' m'
  },
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 300,
    max: 5000,
    step: 100,
    default: 2500,
    unit: ' m'
  }
];
