// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUMultipleTestingAdjustment's `method`, plus the p-values it adjusts.
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
  {id: 'signalShare', label: 'Data: signalShare', min: 0, max: 0.6, step: 0.05, default: 0.25},
  {id: 'testCount', label: 'Data: testCount', min: 20, max: 400, step: 20, default: 200}
];
