// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The staypoint thresholds of GPUTrajectoryStaypoints and the radius of GPUTrajectoryLocations feed
// the indicators; all three are per-frame. World units and playback seconds (a vessel hour is 25 s).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'distanceThreshold',
    label: 'distanceThreshold',
    min: 0.002,
    max: 0.03,
    step: 0.001,
    default: 0.006
  },
  {
    id: 'timeThreshold',
    label: 'timeThreshold',
    min: 0,
    max: 60,
    step: 2,
    default: 12,
    unit: 'playback s'
  },
  {id: 'epsilon', label: 'epsilon', min: 0.002, max: 0.06, step: 0.002, default: 0.02}
];
