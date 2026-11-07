// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to makeGPURasterMetadata({width, height, cellSize, origin, rowDirection}).
export const parameters: GpuSceneParameter[] = [
  {id: 'width', label: 'width', min: 6, max: 48, step: 1, default: 24, unit: 'px'},
  {id: 'height', label: 'height', min: 6, max: 30, step: 1, default: 14, unit: 'px'},
  {id: 'cellSize', label: 'cellSize', min: 0.02, max: 0.07, step: 0.002, default: 0.045},
  {id: 'originX', label: 'origin[0]', min: 0, max: 1.2, step: 0.02, default: 0.3},
  {id: 'originY', label: 'origin[1]', min: 0.3, max: 1, step: 0.02, default: 0.85},
  {
    id: 'rowDirection',
    label: 'rowDirection',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['south', 'north']
  }
];
