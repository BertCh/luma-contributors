// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTrajectoryResample's `sampleCount` and `spacing` props.
export const parameters: GpuSceneParameter[] = [
  {id: 'sampleCount', label: 'sampleCount', min: 2, max: 24, step: 1, default: 10},
  {
    id: 'spacing',
    label: 'spacing',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['time', 'arc-length']
  }
];
