// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialMarkov's `classCount`, `maximumLagClassCount` and `normalize`.
export const parameters: GpuSceneParameter[] = [
  {id: 'classCount', label: 'classCount', min: 3, max: 5, step: 1, default: 4},
  {id: 'maximumLagClassCount', label: 'maximumLagClassCount', min: 2, max: 4, step: 1, default: 4},
  {
    id: 'normalize',
    label: 'normalize',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  },
  {id: 'period', label: 'Show: period', min: 0, max: 5, step: 1, default: 0}
];
