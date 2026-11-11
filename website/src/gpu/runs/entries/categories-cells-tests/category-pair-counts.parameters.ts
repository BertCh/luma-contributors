// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `categoryCount` is a compile-time prop of GPUCategoryPairCounts; the variable only changes which
// tract column is cut into quantile classes.
export const parameters: GpuSceneParameter[] = [
  {id: 'categoryCount', label: 'categoryCount', min: 2, max: 4, step: 1, default: 3},
  {
    id: 'variable',
    label: 'Data: variable',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['poverty', 'unemployment', 'age 65+']
  }
];
