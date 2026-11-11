// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// All per-frame: the slice time, the temporal bandwidth and the spatial bandwidth of
// getGPUSpaceTimeKernelDensityParameterValues.

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'firstSliceTime',
    label: 'firstSliceTime',
    min: 0,
    max: 18,
    step: 0.25,
    default: 9,
    unit: 'h'
  },
  {
    id: 'temporalBandwidth',
    label: 'temporalBandwidth',
    min: 0.25,
    max: 4,
    step: 0.25,
    default: 1,
    unit: 'h'
  },
  {id: 'bandwidth', label: 'bandwidth', min: 0.01, max: 0.08, step: 0.005, default: 0.03}
];
