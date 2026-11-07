// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame parameters of GPUCompositeScore: `scaler`, `aggregation` and one weight.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'scaler',
    label: 'scaler',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['min-max', 'z-score', 'rank']
  },
  {
    id: 'aggregation',
    label: 'aggregation',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['weighted-sum', 'weighted-geometric-mean', 'principal-component']
  },
  {id: 'weight', label: 'weights[1]', min: 0, max: 4, step: 0.25, default: 1}
];
