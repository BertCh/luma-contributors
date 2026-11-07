// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUGammaStatistic's `operation`, plus the data it is computed from.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['cross-product', 'squared-difference', 'absolute-difference']
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.2}
];
