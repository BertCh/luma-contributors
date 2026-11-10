// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUKnoxTest's per-frame `parameters` (time threshold, seed, permutation count), plus
// the data: how far apart events count as spatial neighbors and how clustered they are in time.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'timeThreshold',
    label: 'timeThreshold',
    min: 1,
    max: 30,
    step: 1,
    default: 7,
    unit: ' days'
  },
  {id: 'permutationCount', label: 'permutationCount', min: 19, max: 499, step: 20, default: 199},
  {id: 'radius', label: 'radius', min: 0.5, max: 8, step: 0.5, default: 1, unit: ' km'},
  {id: 'clustering', label: 'Observed time share', min: 0, max: 1, step: 0.05, default: 1}
];
