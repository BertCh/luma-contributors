// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The partition under evaluation comes from GPUWardRegions (`targetRegionCount`, `standardize`).
export const parameters: GpuSceneParameter[] = [
  {id: 'regionCount', label: 'Ward targetRegionCount', min: 2, max: 12, step: 1, default: 5},
  {
    id: 'standardize',
    label: 'Ward standardize',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  }
];
