// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// bandwidth and kernel are written into GPUGeographicallyWeightedRegression's per-frame parameters;
// drift is how strongly the true slope of the first predictor changes from west to east.
export const parameters: GpuSceneParameter[] = [
  {id: 'bandwidth', label: 'bandwidth', min: 0.15, max: 1.2, step: 0.05, default: 0.4},
  {
    id: 'kernel',
    label: 'kernel',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['gaussian', 'bisquare']
  },
  {id: 'drift', label: 'Data: slope drift', min: 0, max: 3, step: 0.1, default: 2.5}
];
