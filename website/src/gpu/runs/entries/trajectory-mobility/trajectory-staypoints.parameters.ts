// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The two per-frame thresholds of GPUTrajectoryStaypoints. Distances are world units (the Bay frame
// is about 25 km across); times are playback seconds (a vessel hour is 25 s).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'distanceThreshold',
    label: 'distanceThreshold',
    min: 0.002,
    max: 0.05,
    step: 0.001,
    default: 0.01
  },
  {
    id: 'timeThreshold',
    label: 'timeThreshold',
    min: 0,
    max: 150,
    step: 5,
    default: 20,
    unit: 'playback s'
  }
];
