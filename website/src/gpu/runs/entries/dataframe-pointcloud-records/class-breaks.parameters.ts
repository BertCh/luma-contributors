// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame settings of GPUClassBreaks (`method`, `classCount`) and the input data.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 4,
    step: 1,
    default: 3,
    options: ['equal-interval', 'quantile', 'standard-deviation', 'natural-breaks', 'head-tail']
  },
  {id: 'classCount', label: 'classCount', min: 2, max: 9, step: 1, default: 5},
  {id: 'spread', label: 'Data: spread', min: 0.5, max: 1.8, step: 0.05, default: 1}
];
