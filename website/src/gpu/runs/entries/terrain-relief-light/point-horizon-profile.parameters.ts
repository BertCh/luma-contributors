// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointHorizonProfile({observers}) and getGPUPointHorizonProfileParameterValues({maximumDistance}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'observerColumn',
    label: 'observers: column',
    min: 0,
    max: 100,
    step: 1,
    default: 40,
    unit: '%'
  },
  {id: 'observerRow', label: 'observers: row', min: 0, max: 100, step: 1, default: 50, unit: '%'},
  {
    id: 'observerHeight',
    label: 'observers: height',
    min: 0,
    max: 100,
    step: 2,
    default: 4,
    unit: ' m'
  },
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 300,
    max: 3600,
    step: 100,
    default: 3000,
    unit: ' m'
  }
];
