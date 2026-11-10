// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUMultivariateLocalGeary's `variableCount`, plus the data it is computed from.
export const parameters: GpuSceneParameter[] = [
  {id: 'variableCount', label: 'variableCount', min: 1, max: 6, step: 1, default: 3},
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.2},
  {id: 'radius', label: 'radius', min: 0.04, max: 0.12, step: 0.005, default: 0.07}
];
