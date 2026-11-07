// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// method is a compile-time prop of addSegregationCounterfactualRecipe; the other sliders change
// the final-period populations.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['composition', 'dual-composition', 'share']
  },
  {id: 'sorting', label: 'Data: sorting', min: 0, max: 1, step: 0.05, default: 0.6},
  {id: 'growth', label: 'Data: groupGrowth', min: 0, max: 1, step: 0.05, default: 0.4}
];
