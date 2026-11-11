// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `alpha` is the per-frame level packed by getGPURasterFalseDiscoveryRateMaskParameterValues;
// `method` is a compile-time prop of GPURasterFalseDiscoveryRateMask.
export const parameters: GpuSceneParameter[] = [
  {id: 'alpha', label: 'alpha', min: 0.01, max: 0.5, step: 0.01, default: 0.2},
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 3,
    step: 1,
    default: 1,
    options: ['benjamini-hochberg', 'benjamini-yekutieli', 'holm', 'bonferroni']
  }
];
