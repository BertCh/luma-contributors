// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialCrossValidatedOLS's `folds.foldCount` and `folds.bufferDistance`, plus
// the noise in the response.
export const parameters: GpuSceneParameter[] = [
  {id: 'foldCount', label: 'foldCount', min: 2, max: 6, step: 1, default: 4},
  {id: 'bufferDistance', label: 'bufferDistance', min: 0, max: 0.15, step: 0.01, default: 0.05},
  {id: 'noise', label: 'Data: noise', min: 0, max: 0.5, step: 0.02, default: 0}
];
