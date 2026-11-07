// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainCriticalPoints({connectivity}); noise is added to the input elevation.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'connectivity',
    label: 'connectivity',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['8', '6']
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 6, step: 0.5, default: 2, unit: ' m'}
];
