// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUProfilePeaks({window, suppressionRadius}) and getGPUProfilePeaksParameterValues({minimumProminence}).
export const parameters: GpuSceneParameter[] = [
  {id: 'minimumProminence', label: 'minimumProminence', min: 0, max: 2, step: 0.05, default: 0.5},
  {id: 'window', label: 'window', min: 2, max: 16, step: 1, default: 6, unit: ' samples'},
  {
    id: 'suppressionRadius',
    label: 'suppressionRadius',
    min: 1,
    max: 12,
    step: 1,
    default: 4,
    unit: ' samples'
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.3}
];
