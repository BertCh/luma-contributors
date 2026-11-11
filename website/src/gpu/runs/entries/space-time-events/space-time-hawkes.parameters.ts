// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// maximumLag and maximumDistance are per-frame (the parameter view); maximumIterations is
// compile-time (the EM iterations unrolled into the graph).

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'maximumLag', label: 'maximumLag', min: 1, max: 60, step: 1, default: 30, unit: 'd'},
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 10,
    max: 300,
    step: 10,
    default: 150,
    unit: 'km'
  },
  {id: 'maximumIterations', label: 'maximumIterations', min: 1, max: 60, step: 1, default: 40}
];
