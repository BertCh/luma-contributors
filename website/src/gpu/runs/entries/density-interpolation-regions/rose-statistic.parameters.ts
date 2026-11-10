// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPURoseStatistic's `circular` contract (`mode`, `binCount`); the data filter removes short legs in scene units.
export const parameters: GpuSceneParameter[] = [
  {id: 'binCount', label: 'circular.binCount', min: 6, max: 36, step: 2, default: 18},
  {
    id: 'mode',
    label: 'circular.mode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['directional', 'axial']
  },
  {
    id: 'minimumLength',
    label: 'Data: minimum leg length',
    min: 0,
    max: 0.04,
    step: 0.002,
    default: 0
  }
];
