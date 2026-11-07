// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUWardRegions' `targetRegionCount`, `mergeSchedule` and `standardize`.
export const parameters: GpuSceneParameter[] = [
  {id: 'targetRegionCount', label: 'targetRegionCount', min: 2, max: 12, step: 1, default: 6},
  {
    id: 'mergeSchedule',
    label: 'mergeSchedule',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['sequential', 'mutual-nearest']
  },
  {
    id: 'standardize',
    label: 'standardize',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  }
];
