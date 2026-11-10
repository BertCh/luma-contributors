// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudHeight's `interpolation` and `grid` (cell size), plus point noise.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'interpolation',
    label: 'interpolation',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['default (cell)', 'bilinear', 'idw']
  },
  {id: 'cellSize', label: 'grid.cellSize', min: 0.02, max: 0.1, step: 0.01, default: 0.05},
  {id: 'noise', label: 'Data: jitter', min: 0, max: 0.03, step: 0.002, default: 0}
];
