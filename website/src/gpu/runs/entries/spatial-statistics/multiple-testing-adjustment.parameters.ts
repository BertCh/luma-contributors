// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUMultipleTestingAdjustment's `method`, the significance level the adjusted p-values
// are read against, and the p-values it adjusts.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 4,
    step: 1,
    default: 3,
    options: ['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli']
  },
  {id: 'alpha', label: 'Show: alpha', min: 0.01, max: 0.2, step: 0.01, default: 0.05},
  {id: 'testCount', label: 'Data: tract count', min: 20, max: 150, step: 10, default: 120}
];
