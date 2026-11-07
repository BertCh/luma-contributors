// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPURasterProfileParameterValues({spacing, method}, grid).
export const parameters: GpuSceneParameter[] = [
  {id: 'spacing', label: 'spacing', min: 0.008, max: 0.1, step: 0.002, default: 0.03},
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['nearest', 'bilinear', 'bicubic']
  },
  {id: 'gridWidth', label: 'grid.width', min: 16, max: 96, step: 4, default: 40, unit: 'cells'}
];
