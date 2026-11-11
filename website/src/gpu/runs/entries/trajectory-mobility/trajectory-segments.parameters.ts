// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The two per-frame settings of GPUTrajectorySegments, in playback seconds (a vessel hour is 25 s).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'activityMinimumDuration',
    label: 'activityMinimumDuration',
    min: 0,
    max: 150,
    step: 5,
    default: 40,
    unit: 'playback s'
  },
  {id: 'maximumGap', label: 'maximumGap', min: 4, max: 40, step: 2, default: 16, unit: 'playback s'}
];
