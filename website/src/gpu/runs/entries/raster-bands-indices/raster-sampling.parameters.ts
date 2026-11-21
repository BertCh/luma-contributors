// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPURasterSamplingParameterValues({method, noDataPolicy}, grid).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['nearest', 'bilinear', 'bicubic']
  },
  {
    id: 'noDataPolicy',
    label: 'noDataPolicy',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['propagate', 'renormalize']
  },
  {id: 'gridWidth', label: 'grid.width', min: 8, max: 40, step: 2, default: 16, unit: 'cells'}
];
