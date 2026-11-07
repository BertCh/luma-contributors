// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUContinuousSpatialSample's `spacing` and `seed`.
export const parameters: GpuSceneParameter[] = [
  {id: 'spacing', label: 'spacing', min: 0.03, max: 0.12, step: 0.005, default: 0.06},
  {id: 'seed', label: 'seed', min: 1, max: 30, step: 1, default: 4}
];
