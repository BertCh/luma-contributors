// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// hops and startCount are GPUNetworkNeighborhood's per-frame views.
export const parameters: GpuSceneParameter[] = [
  {id: 'hops', label: 'hops', min: 1, max: 12, step: 1, default: 7},
  {id: 'startCount', label: 'startCount', min: 1, max: 4, step: 1, default: 2}
];
