// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudSurfaces' `groundClass` and `grid` (cell size, compile-time).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'groundClass',
    label: 'groundClass',
    min: 1,
    max: 2,
    step: 1,
    default: 2,
    options: ['1 (unclassified)', '2 (ground)']
  },
  {id: 'cellSize', label: 'grid.cellSize', min: 0.02, max: 0.1, step: 0.01, default: 0.04},
  {id: 'noise', label: 'Data: jitter', min: 0, max: 0.03, step: 0.002, default: 0}
];
