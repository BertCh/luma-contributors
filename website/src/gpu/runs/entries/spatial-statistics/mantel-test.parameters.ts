// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUMantelTest's per-frame `parameters` (seed, permutation count), plus the data:
// how far apart events count as spatial neighbors and how clustered they are in time.
export const parameters: GpuSceneParameter[] = [
  {id: 'permutationCount', label: 'permutationCount', min: 19, max: 499, step: 20, default: 199},
  {id: 'seed', label: 'seed', min: 1, max: 40, step: 1, default: 3},
  {id: 'radius', label: 'Data: radius', min: 0.03, max: 0.12, step: 0.005, default: 0.06},
  {id: 'clustering', label: 'Data: clustering', min: 0, max: 1, step: 0.05, default: 0.7}
];
