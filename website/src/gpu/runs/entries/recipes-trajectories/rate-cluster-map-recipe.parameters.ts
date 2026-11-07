// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// analyze is a compile-time prop of addRateClusterMapRecipe; significanceLevel is one of its
// per-frame settings; the small-zone share shapes the input counts.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'analyze',
    label: 'analyze',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['standardized', 'smoothed']
  },
  {
    id: 'significanceLevel',
    label: 'significanceLevel',
    min: 0.01,
    max: 0.4,
    step: 0.01,
    default: 0.1
  },
  {id: 'smallShare', label: 'Data: smallZoneShare', min: 0, max: 0.8, step: 0.05, default: 0.3}
];
