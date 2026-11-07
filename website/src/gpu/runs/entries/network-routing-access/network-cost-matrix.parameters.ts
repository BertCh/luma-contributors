// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// costLimit is GPUNetworkCostMatrix's per-frame view; origins and destinations size its inputs.
export const parameters: GpuSceneParameter[] = [
  {id: 'costLimit', label: 'costLimit', min: 0.1, max: 2, step: 0.05, default: 1.6},
  {id: 'startNodes', label: 'startNodes', min: 3, max: 14, step: 1, default: 10},
  {id: 'targetNodes', label: 'targetNodes', min: 3, max: 14, step: 1, default: 14},
  {id: 'congestion', label: 'Data: congestion', min: 0, max: 3, step: 0.1, default: 0.6}
];
