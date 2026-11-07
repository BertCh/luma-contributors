// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// costLimit, decay and beta are GPUNetworkCostWeights' per-frame `parameters` words.
export const parameters: GpuSceneParameter[] = [
  {id: 'costLimit', label: 'costLimit', min: 0.2, max: 1.6, step: 0.05, default: 1.2},
  {
    id: 'decay',
    label: 'decay',
    min: 0,
    max: 3,
    step: 1,
    default: 1,
    options: ['none', 'exponential', 'power', 'gaussian']
  },
  {id: 'beta', label: 'beta', min: 0.5, max: 6, step: 0.25, default: 3}
];
