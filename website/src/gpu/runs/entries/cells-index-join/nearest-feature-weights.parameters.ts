// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `k`, `weightType` and `power` are GPUNearestFeatureWeights props (compile time). Dots are sized
// by nothing but colored by each point's total weight (the row sum).
export const parameters: GpuSceneParameter[] = [
  {id: 'k', label: 'k', min: 1, max: 5, step: 1, default: 3},
  {
    id: 'weightType',
    label: 'weightType',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['binary', 'inverse-distance']
  },
  {id: 'power', label: 'power', min: 1, max: 3, step: 1, default: 1}
];
