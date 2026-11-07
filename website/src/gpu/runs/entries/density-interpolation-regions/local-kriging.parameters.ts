// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPULocalKriging's `mode`, `neighborCount` and `covariance` props.
export const parameters: GpuSceneParameter[] = [
  {id: 'mode', label: 'mode', min: 0, max: 1, step: 1, default: 1, options: ['simple', 'ordinary']},
  {id: 'range', label: 'covariance.range', min: 0.05, max: 0.6, step: 0.01, default: 0.3},
  {id: 'neighborCount', label: 'neighborCount', min: 4, max: 24, step: 2, default: 12},
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['prediction', 'variance']
  }
];
