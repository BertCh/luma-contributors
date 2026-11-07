// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialCrossValidation's `foldCount`, `seed` and `bufferDistance`.
export const parameters: GpuSceneParameter[] = [
  {id: 'foldCount', label: 'foldCount', min: 2, max: 6, step: 1, default: 4},
  {id: 'seed', label: 'seed', min: 1, max: 40, step: 1, default: 7},
  {id: 'bufferDistance', label: 'bufferDistance', min: 0, max: 0.15, step: 0.01, default: 0.06}
];
