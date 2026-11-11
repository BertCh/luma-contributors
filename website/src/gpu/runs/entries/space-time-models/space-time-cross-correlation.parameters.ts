// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `lags` is the compile-time list -k..k; `multipleTesting` is compile-time (GPUMultipleTestingMethod);
// `significanceLevel` is the adjusted p-value below which a cell's best lag is mapped.
export const parameters: GpuSceneParameter[] = [
  {id: 'lags', label: 'lags', min: 1, max: 4, step: 1, default: 2, unit: '± periods'},
  {
    id: 'multipleTesting',
    label: 'multipleTesting',
    min: 0,
    max: 4,
    step: 1,
    default: 3,
    options: ['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli']
  },
  {
    id: 'significanceLevel',
    label: 'significanceLevel',
    min: 0.01,
    max: 0.2,
    step: 0.01,
    default: 0.05
  }
];
