// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpaceTimeLocalMoran's `period`, `variant` and `lag` (compile-time) and the per-frame
// permutation count. The cube has 9 periods; `period` must be at least `lag`.
export const parameters: GpuSceneParameter[] = [
  {id: 'period', label: 'period', min: 3, max: 8, step: 1, default: 5},
  {
    id: 'variant',
    label: 'variant',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['lagged', 'differential']
  },
  {id: 'lag', label: 'lag', min: 1, max: 3, step: 1, default: 1},
  {id: 'permutationCount', label: 'permutationCount', min: 19, max: 499, step: 20, default: 199}
];
