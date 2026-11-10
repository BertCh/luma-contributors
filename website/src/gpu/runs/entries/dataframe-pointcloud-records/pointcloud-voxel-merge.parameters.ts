// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The voxel size of the two tile reductions whose `parts` GPUPointCloudVoxelMerge combines
// (compile-time), and where the seam between the two tiles lies (per frame).
export const parameters: GpuSceneParameter[] = [
  {id: 'voxelSize', label: 'Parts: voxelSize', min: 0.04, max: 0.2, step: 0.01, default: 0.08},
  {
    id: 'tileBoundary',
    label: 'tile boundary',
    min: 0.2,
    max: 1.6,
    step: 0.02,
    default: 0.9
  }
];
