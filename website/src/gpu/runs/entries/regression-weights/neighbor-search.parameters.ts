// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUNeighborSearch's `mode` and `k` props and the per-frame `radius` parameter.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'mode',
    label: 'mode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['knn', 'radius']
  },
  {id: 'k', label: 'k', min: 1, max: 8, step: 1, default: 4},
  {
    id: 'radius',
    label: 'radius',
    min: 0.03,
    max: 0.2,
    step: 0.005,
    default: 0.09
  }
];
