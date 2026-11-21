// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// All three are compile-time props of GPUModalSplit: the two finite entries of modeSpeedUpperBoundsFloat32
// (world units per playback second; the third is Infinity) and the metric.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'slowBreak',
    label: 'modeSpeedUpperBoundsFloat32[0]',
    min: 0.004,
    max: 0.016,
    step: 0.004,
    default: 0.012
  },
  {
    id: 'fastBreak',
    label: 'modeSpeedUpperBoundsFloat32[1]',
    min: 0.02,
    max: 0.05,
    step: 0.005,
    default: 0.03
  },
  {
    id: 'metric',
    label: 'metric',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['count', 'distance', 'duration']
  }
];
