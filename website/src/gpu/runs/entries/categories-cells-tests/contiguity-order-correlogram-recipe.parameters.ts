// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `maximumOrder` is a compile-time prop of addContiguityOrderCorrelogramRecipe. The variable only
// chooses which tract column Moran's I is computed on.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumOrder', label: 'maximumOrder', min: 2, max: 6, step: 1, default: 5},
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
