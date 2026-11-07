// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// selectCount is compile-time; threshold is GPUNetworkFacilityOptimization's per-frame view.
export const parameters: GpuSceneParameter[] = [
  {id: 'selectCount', label: 'selectCount', min: 2, max: 8, step: 1, default: 4},
  {id: 'threshold', label: 'threshold', min: 0.2, max: 1.2, step: 0.05, default: 0.6}
];
