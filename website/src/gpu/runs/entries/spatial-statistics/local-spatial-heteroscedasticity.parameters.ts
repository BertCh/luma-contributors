// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPULocalSpatialHeteroscedasticity's `residualPower`, plus the data it is computed from.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'residualPower',
    label: 'residualPower',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['1', '2']
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.7},
  {id: 'radius', label: 'radius', min: 0.04, max: 0.12, step: 0.005, default: 0.07}
];
