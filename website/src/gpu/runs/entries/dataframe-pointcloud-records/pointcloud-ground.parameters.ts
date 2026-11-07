// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudGround's `radii` (window count), `initialThreshold`, `slope` and
// `maximumThreshold`; all are compile-time.
export const parameters: GpuSceneParameter[] = [
  {id: 'radiusCount', label: 'radii (windows)', min: 1, max: 4, step: 1, default: 3},
  {
    id: 'initialThreshold',
    label: 'initialThreshold',
    min: 0.005,
    max: 0.1,
    step: 0.005,
    default: 0.02
  },
  {id: 'slope', label: 'slope', min: 0.01, max: 0.5, step: 0.01, default: 0.1},
  {
    id: 'maximumThreshold',
    label: 'maximumThreshold',
    min: 0.02,
    max: 0.3,
    step: 0.01,
    default: 0.08
  }
];
