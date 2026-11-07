// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// threshold, decay and beta are GPUNetworkAccessibility's per-frame `parameters` words.
export const parameters: GpuSceneParameter[] = [
  {id: 'threshold', label: 'threshold', min: 0.1, max: 2.5, step: 0.05, default: 0.8},
  {
    id: 'decay',
    label: 'decay',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['none', 'exponential', 'power']
  },
  {id: 'beta', label: 'beta', min: 0.2, max: 6, step: 0.2, default: 2},
  {
    id: 'measure',
    label: 'Output: measure',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['gravity', 'cumulative']
  }
];
