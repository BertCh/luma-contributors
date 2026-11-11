// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `miscoverage` (set by the pointer's rank rule) is the per-frame alpha of getGPUConformalPredictionParameterValues. `mask` keeps
// that share of the calibration rows (the operation's calibration.mask). `scales` supplies or
// omits the heteroscedastic scales (calibration.scales and query.scales), fixed at construction.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'miscoverage',
    label: 'miscoverage',
    min: 0.02,
    max: 0.5,
    step: 0.01,
    default: 0.1
  },
  {
    id: 'mask',
    label: 'mask',
    min: 10,
    max: 100,
    step: 5,
    default: 100,
    unit: '%'
  },
  {
    id: 'scales',
    label: 'scales',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['omitted', 'local spread']
  }
];
