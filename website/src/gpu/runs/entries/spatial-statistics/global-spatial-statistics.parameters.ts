// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUGlobalSpatialStatistics's `alternative`, plus the data it is computed from.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'alternative',
    label: 'alternative',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['two-sided', 'greater', 'lesser']
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.3},
  {id: 'radius', label: 'Data: radius', min: 0.04, max: 0.12, step: 0.005, default: 0.07}
];
