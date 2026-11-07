// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// bandwidth is a compile-time prop of GPUNetworkKernelDensity; clustering moves the events.
export const parameters: GpuSceneParameter[] = [
  {id: 'bandwidth', label: 'bandwidth', min: 0.1, max: 1, step: 0.05, default: 0.4},
  {id: 'clustering', label: 'Data: clustering', min: 0, max: 1, step: 0.05, default: 0.7}
];
