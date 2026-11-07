// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSkaterRegions' per-frame `regionCount` and `minimumRegionSize` (its parameters view).
export const parameters: GpuSceneParameter[] = [
  {id: 'regionCount', label: 'regionCount', min: 2, max: 12, step: 1, default: 6},
  {id: 'minimumRegionSize', label: 'minimumRegionSize', min: 1, max: 12, step: 1, default: 3}
];
