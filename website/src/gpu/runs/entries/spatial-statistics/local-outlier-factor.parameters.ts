// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPULocalOutlierFactor's per-frame `threshold`, plus the neighbor graph it reads.
export const parameters: GpuSceneParameter[] = [
  {id: 'threshold', label: 'threshold', min: 1, max: 3, step: 0.05, default: 1.5},
  {id: 'neighborCount', label: 'Data: neighborCount', min: 4, max: 30, step: 1, default: 12}
];
