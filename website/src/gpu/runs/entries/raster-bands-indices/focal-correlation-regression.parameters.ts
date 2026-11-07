// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'metric',
    label: 'Output shown',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['slope', 'correlation', 'rSquared', 'intercept']
  },
  {id: 'radius', label: 'radius', min: 1, max: 8, step: 1, default: 4, unit: 'cells'},
  {id: 'minimumCount', label: 'minimumCount', min: 2, max: 40, step: 1, default: 6},
  {id: 'noise', label: 'Data: noise', min: 0, max: 0.4, step: 0.02, default: 0.1}
];
