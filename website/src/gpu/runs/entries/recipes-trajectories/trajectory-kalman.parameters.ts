// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the accelerationDensity and measurementVariance entries of GPUTrajectoryKalman's
// parameters buffer; the noise slider only changes how jittery the input fixes are.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'accelerationDensity',
    label: 'accelerationDensity',
    min: 0.1,
    max: 40,
    step: 0.1,
    default: 4,
    unit: 'x1e-9'
  },
  {
    id: 'measurementVariance',
    label: 'measurementVariance',
    min: 20,
    max: 2500,
    step: 20,
    default: 600,
    unit: 'x1e-6'
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 0.06, step: 0.002, default: 0.026}
];
