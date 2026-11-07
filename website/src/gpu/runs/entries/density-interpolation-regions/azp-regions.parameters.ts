// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUAZPRegions' `regionCapacity`, `maximumIterations` and the seed in `parameters`.
export const parameters: GpuSceneParameter[] = [
  {id: 'regionCapacity', label: 'regionCapacity', min: 2, max: 10, step: 1, default: 6},
  {id: 'maximumIterations', label: 'maximumIterations', min: 1, max: 61, step: 2, default: 31},
  {id: 'seed', label: 'seed', min: 1, max: 20, step: 1, default: 3}
];
