// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the destinationGrid affine (rotation, scale) and method.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'rotation',
    label: 'destinationGrid rotation',
    min: -90,
    max: 90,
    step: 1,
    default: 25,
    unit: '°'
  },
  {id: 'scale', label: 'destinationGrid cell size', min: 0.5, max: 1.5, step: 0.05, default: 1},
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['nearest', 'bilinear']
  }
];
