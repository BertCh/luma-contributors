// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// family and kernel are GPUGeneralizedGeographicallyWeightedRegression props; bandwidth is the first
// entry of its per-frame parameters; drift is how strongly the true slope changes from west to east.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'family',
    label: 'family',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['poisson', 'logistic']
  },
  {
    id: 'kernel',
    label: 'kernel',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['gaussian', 'bisquare']
  },
  {id: 'bandwidth', label: 'bandwidth', min: 0.2, max: 1.2, step: 0.05, default: 0.5},
  {id: 'drift', label: 'Data: slope drift', min: 0, max: 2, step: 0.1, default: 1.5}
];
