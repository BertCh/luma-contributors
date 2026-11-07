// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudFeatures' compile-time `capacity`, which feature is drawn, and point noise.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'shape',
    label: 'shapes',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['linearity', 'planarity', 'scattering']
  },
  {id: 'capacity', label: 'capacity', min: 8, max: 32, step: 2, default: 20},
  {id: 'noise', label: 'Data: jitter', min: 0, max: 0.01, step: 0.0005, default: 0.0005}
];
