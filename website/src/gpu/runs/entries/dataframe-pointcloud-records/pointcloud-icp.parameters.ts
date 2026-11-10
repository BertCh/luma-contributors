// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudICP's `iterations` and `maximumDistance` (compile-time), and the rotation
// and shift that misalign the source cloud (per frame).
export const parameters: GpuSceneParameter[] = [
  {id: 'iterations', label: 'iterations', min: 1, max: 20, step: 1, default: 12},
  {id: 'maximumDistance', label: 'maximumDistance', min: 0.03, max: 0.4, step: 0.01, default: 0.2},
  {
    id: 'rotation',
    label: 'misalignment rotation',
    min: -20,
    max: 20,
    step: 1,
    default: 8,
    unit: 'deg'
  },
  {id: 'shift', label: 'misalignment shift', min: 0, max: 0.08, step: 0.004, default: 0.036}
];
