// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'mergeThreshold', label: 'mergeThreshold', min: 0, max: 0.6, step: 0.02, default: 0.2},
  {
    id: 'mergeMode',
    label: 'mergeMode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['fixed', 'adaptive']
  },
  {id: 'maximumIterations', label: 'maximumIterations', min: 1, max: 24, step: 1, default: 8}
];
