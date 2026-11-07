// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// costLimit is GPUNetworkCostTable's per-frame radius view.
export const parameters: GpuSceneParameter[] = [
  {id: 'costLimit', label: 'costLimit', min: 0.05, max: 0.4, step: 0.01, default: 0.22},
  {id: 'congestion', label: 'Data: congestion', min: 0, max: 3, step: 0.1, default: 0.6}
];
