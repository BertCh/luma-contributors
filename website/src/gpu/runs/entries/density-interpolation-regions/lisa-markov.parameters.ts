// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPULISAMarkov's `gating`, `periodLag` and the significance level in `parameters`.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'gating',
    label: 'gating',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['significance', 'none']
  },
  {
    id: 'significanceLevel',
    label: 'significanceLevel',
    min: 0.01,
    max: 0.3,
    step: 0.01,
    default: 0.1
  },
  {id: 'periodLag', label: 'periodLag', min: 1, max: 3, step: 1, default: 1},
  {id: 'period', label: 'Show: period', min: 0, max: 5, step: 1, default: 0}
];
