// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUGlobalPermutationTest's `statistic` and its per-frame `parameters` (seed and
// permutation count), plus the data noise.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'statistic',
    label: 'statistic',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['moran', 'geary', 'getisOrdG']
  },
  {id: 'permutationCount', label: 'permutationCount', min: 49, max: 999, step: 50, default: 999},
  {id: 'seed', label: 'seed', min: 1, max: 40, step: 1, default: 3},
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.75}
];
