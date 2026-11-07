// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUNestedInterpolationValidation's `foldCount`, `innerFoldCount` and
// `neighborCount`, plus the noise in the response.
export const parameters: GpuSceneParameter[] = [
  {id: 'foldCount', label: 'foldCount', min: 2, max: 5, step: 1, default: 3},
  {id: 'innerFoldCount', label: 'innerFoldCount', min: 2, max: 4, step: 1, default: 3},
  {id: 'neighborCount', label: 'neighborCount', min: 8, max: 32, step: 4, default: 16},
  {id: 'noise', label: 'Data: noise', min: 0, max: 0.4, step: 0.02, default: 0.08}
];
