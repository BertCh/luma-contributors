// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudChange's `radius` and `halfDepth` (compile-time) and the size of the
// simulated change between the two epochs.
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 0.03, max: 0.15, step: 0.01, default: 0.07},
  {id: 'halfDepth', label: 'halfDepth', min: 0.02, max: 0.3, step: 0.02, default: 0.12},
  {id: 'change', label: 'Data: slump depth', min: 0, max: 0.12, step: 0.01, default: 0.06}
];
