// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudSMRF's `slope`, `threshold` and `cut` (compile-time).
export const parameters: GpuSceneParameter[] = [
  {id: 'threshold', label: 'threshold', min: 0.01, max: 0.3, step: 0.01, default: 0.04},
  {id: 'slope', label: 'slope', min: 0.02, max: 0.6, step: 0.02, default: 0.15},
  {id: 'cut', label: 'cut', min: 0, max: 0.4, step: 0.04, default: 0}
];
