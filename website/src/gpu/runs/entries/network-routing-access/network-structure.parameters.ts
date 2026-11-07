// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The output picks which GPUNetworkStructure column is drawn; closures thin the network.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['bridges', 'articulationPoints', 'strongComponents']
  },
  {id: 'closures', label: 'Data: closures', min: 0, max: 0.6, step: 0.05, default: 0.4}
];
