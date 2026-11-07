// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// degreeBinning is a compile-time prop of GPUNetworkStatistics; closures thin the network.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'degreeBinning',
    label: 'degreeBinning',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['linear', 'log2']
  },
  {id: 'closures', label: 'Data: closures', min: 0, max: 0.7, step: 0.05, default: 0.25}
];
