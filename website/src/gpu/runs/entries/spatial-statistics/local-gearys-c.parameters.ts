// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPULocalGeary's `islands` prop, plus the band radius of its weights and the data the statistic is computed from.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'islands',
    label: 'islands',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['exclude', 'keep']
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.15},
  {id: 'radius', label: 'radius', min: 0.03, max: 0.12, step: 0.005, default: 0.05}
];
