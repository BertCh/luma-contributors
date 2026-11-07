// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `order` is a GPUHilbertKeys prop (curve resolution, compile time). `jitter` moves the points per
// frame and the keys are recomputed.
export const parameters: GpuSceneParameter[] = [
  {id: 'order', label: 'order', min: 1, max: 6, step: 1, default: 3},
  {id: 'jitter', label: 'Data: jitter', min: 0, max: 0.12, step: 0.005, default: 0.03}
];
