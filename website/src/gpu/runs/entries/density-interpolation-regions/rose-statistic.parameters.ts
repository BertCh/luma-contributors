// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPURoseStatistic's `circular` contract (`mode`, `binCount`); concentration varies the data.
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
  {id: 'concentration', label: 'Data: spread', min: 0.1, max: 1.2, step: 0.05, default: 0.4}
];
