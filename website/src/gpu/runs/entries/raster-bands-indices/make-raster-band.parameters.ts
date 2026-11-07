// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to makeGPURasterBand(values, grid, {noDataValue, scale, offset}).
export const parameters: GpuSceneParameter[] = [
  {id: 'scale', label: 'scale (x 1e-5)', min: 1.5, max: 5, step: 0.05, default: 2.75},
  {id: 'offset', label: 'offset', min: -0.5, max: 0.1, step: 0.01, default: -0.2},
  {
    id: 'noDataValue',
    label: 'noDataValue',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['unset', '0']
  }
];
