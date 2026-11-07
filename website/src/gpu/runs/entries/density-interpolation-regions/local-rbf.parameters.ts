// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPULocalRBF's `kernel`, `smoothing` and `neighborCount` props.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'kernel',
    label: 'kernel',
    min: 0,
    max: 4,
    step: 1,
    default: 0,
    options: ['thin-plate-spline', 'cubic', 'gaussian', 'linear', 'quintic']
  },
  {id: 'smoothing', label: 'smoothing', min: 0, max: 0.2, step: 0.01, default: 0.01},
  {id: 'neighborCount', label: 'neighborCount', min: 6, max: 32, step: 2, default: 16}
];
