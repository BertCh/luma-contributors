// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPULOSHResidualPermutationTest's `alternative` and its per-frame permutation count.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'alternative',
    label: 'alternative',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['two-sided', 'greater', 'lesser']
  },
  {id: 'permutationCount', label: 'permutationCount', min: 19, max: 499, step: 20, default: 199},
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.9}
];
