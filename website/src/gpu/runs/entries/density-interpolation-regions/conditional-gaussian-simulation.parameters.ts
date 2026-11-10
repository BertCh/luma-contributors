// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUConditionalGaussianSimulation's `seed` and `covariance` props.
export const parameters: GpuSceneParameter[] = [
  {id: 'seed', label: 'seed', min: 1, max: 30, step: 1, default: 4},
  {id: 'replicate', label: 'Show: replicate', min: 0, max: 15, step: 1, default: 0},
  {id: 'range', label: 'covariance.range', min: 0.05, max: 0.6, step: 0.01, default: 0.25},
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['realization', 'mean', 'standard deviation']
  }
];
