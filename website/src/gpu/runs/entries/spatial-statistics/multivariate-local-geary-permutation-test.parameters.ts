// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUMultivariateLocalGearyPermutationTest's `variableCount`, `tail` and
// `permutationCount`, plus the data noise.
export const parameters: GpuSceneParameter[] = [
  {id: 'variableCount', label: 'variableCount', min: 1, max: 5, step: 1, default: 3},
  {
    id: 'tail',
    label: 'tail',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['equal-tail', 'esda-min-tail']
  },
  {
    id: 'permutationCount',
    label: 'permutationCount',
    min: 49,
    max: 499,
    step: 50,
    default: 199
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.3}
];
