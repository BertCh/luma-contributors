// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// distance is the per-frame distance of GPUTrajectoryClosestApproach (world units).
export const parameters: GpuSceneParameter[] = [
  {id: 'distance', label: 'distance', min: 0.01, max: 0.15, step: 0.005, default: 0.05}
];
