// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudGround's `radii` (window count) and `initialThreshold`; both are compile-time.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'initialThreshold',
    label: 'initialThreshold',
    min: 0.005,
    max: 0.8,
    step: 0.005,
    default: 0.02
  },
  {id: 'radiusCount', label: 'radii (windows)', min: 1, max: 4, step: 1, default: 3}
];
