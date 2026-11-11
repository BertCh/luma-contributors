// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The per-frame limits of GPUTrajectoryOutlierFlags: world units per playback second, and per
// second squared. 0 disables a test.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumSpeed', label: 'maximumSpeed', min: 0.005, max: 0.1, step: 0.005, default: 0.04},
  {
    id: 'maximumAcceleration',
    label: 'maximumAcceleration',
    min: 0,
    max: 0.01,
    step: 0.0005,
    default: 0
  }
];
