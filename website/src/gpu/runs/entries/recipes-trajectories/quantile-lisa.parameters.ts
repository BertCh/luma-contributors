// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// numberOfQuantiles and selectedQuantile are the per-frame quantileParameters of GPUQuantileLISA,
// significanceLevel one of the permutation parameters; noise changes the input values.
export const parameters: GpuSceneParameter[] = [
  {id: 'numberOfQuantiles', label: 'numberOfQuantiles', min: 2, max: 8, step: 1, default: 4},
  {id: 'selectedQuantile', label: 'selectedQuantile', min: 1, max: 8, step: 1, default: 4},
  {
    id: 'significanceLevel',
    label: 'significanceLevel',
    min: 0.01,
    max: 0.3,
    step: 0.01,
    default: 0.1
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.3}
];
