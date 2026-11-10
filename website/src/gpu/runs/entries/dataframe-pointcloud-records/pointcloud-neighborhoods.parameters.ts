// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudNeighborhoods' `radius` and `capacity` (both compile-time), plus point noise.
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 0.01, max: 0.08, step: 0.005, default: 0.035},
  {id: 'capacity', label: 'capacity', min: 4, max: 32, step: 2, default: 16},
  {id: 'noise', label: 'Data: jitter', min: 0, max: 0.02, step: 0.002, default: 0}
];
