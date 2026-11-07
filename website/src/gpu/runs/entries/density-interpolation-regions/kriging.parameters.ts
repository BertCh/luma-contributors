// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUKriging's parameter buffer: the variogram (model, range, nugget) and neighborCount.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'model',
    label: 'variogram.model',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['spherical', 'exponential', 'gaussian']
  },
  {id: 'range', label: 'variogram.range', min: 0.05, max: 0.6, step: 0.01, default: 0.3},
  {id: 'nugget', label: 'variogram.nugget', min: 0, max: 0.5, step: 0.01, default: 0.05},
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['values', 'variance']
  }
];
