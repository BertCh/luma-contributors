// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUMaxPRegions' per-frame `threshold`, its `startCount` and the seed in `parameters`.
export const parameters: GpuSceneParameter[] = [
  {id: 'threshold', label: 'threshold', min: 4000, max: 40000, step: 1000, default: 12000},
  {id: 'startCount', label: 'startCount', min: 1, max: 32, step: 1, default: 8},
  {id: 'seed', label: 'seed', min: 1, max: 20, step: 1, default: 3}
];
