// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialWeightsTransform's `operation`, `kernel` and `bandwidth` props.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 5,
    step: 1,
    default: 2,
    options: ['row', 'binary', 'kernel', 'double', 'variance', 'average-reciprocal']
  },
  {
    id: 'kernel',
    label: 'kernel',
    min: 0,
    max: 4,
    step: 1,
    default: 0,
    options: ['gaussian', 'triangular', 'epanechnikov', 'bisquare', 'uniform']
  },
  {
    id: 'bandwidth',
    label: 'bandwidth',
    min: 0.04,
    max: 0.16,
    step: 0.01,
    default: 0.08
  }
];
