// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUColumnQuantiles' `quantileCount` (compile-time) and the per-frame `parameters`:
// the percentile filter range.
export const parameters: GpuSceneParameter[] = [
  {id: 'quantileCount', label: 'quantileCount', min: 2, max: 12, step: 1, default: 4},
  {
    id: 'filterLower',
    label: 'filterRange low',
    min: 0,
    max: 45,
    step: 5,
    default: 10,
    unit: '%'
  },
  {
    id: 'filterUpper',
    label: 'filterRange high',
    min: 55,
    max: 100,
    step: 5,
    default: 90,
    unit: '%'
  }
];
