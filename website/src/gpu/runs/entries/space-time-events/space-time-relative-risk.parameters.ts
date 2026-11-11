// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// All per-frame: the slice time and both bandwidths, packed with
// getGPUSpaceTimeKernelDensityParameterValues for cases and controls alike.

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
    min: 0.5,
    max: 6,
    step: 0.25,
    default: 2,
    unit: 'h'
  },
  {id: 'bandwidth', label: 'bandwidth', min: 0.02, max: 0.1, step: 0.005, default: 0.05}
];
