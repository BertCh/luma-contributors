// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// distance is the per-frame distance of GPUTrajectoryEncounters; bucketCount (columns of the
// resampled table) is compile-time.
export const parameters: GpuSceneParameter[] = [
  {id: 'distance', label: 'distance', min: 0.01, max: 0.12, step: 0.005, default: 0.06},
  {id: 'bucketCount', label: 'bucketCount', min: 4, max: 40, step: 2, default: 24}
];
