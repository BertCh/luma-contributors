// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// `neighborCount` is compile-time in GPUJacquezTest and sizes the spatial list. The permutation count is
// per-frame. The data control exchanges a share of the origin times between events.

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'neighborCount', label: 'Neighbors', min: 2, max: 12, step: 1, default: 5},
  {id: 'permutationCount', label: 'permutationCount', min: 19, max: 499, step: 20, default: 199},
  {id: 'observedShare', label: 'Data: observed times', min: 0, max: 1, step: 0.05, default: 1}
];
