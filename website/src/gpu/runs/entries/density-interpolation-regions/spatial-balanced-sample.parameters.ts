// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialBalancedSample's per-stratum `allocations` and its `seed`.
export const parameters: GpuSceneParameter[] = [
  {id: 'westCount', label: 'allocations[0].sampleCount', min: 10, max: 160, step: 10, default: 60},
  {id: 'eastCount', label: 'allocations[1].sampleCount', min: 10, max: 160, step: 10, default: 100},
  {id: 'seed', label: 'seed', min: 1, max: 30, step: 1, default: 5}
];
