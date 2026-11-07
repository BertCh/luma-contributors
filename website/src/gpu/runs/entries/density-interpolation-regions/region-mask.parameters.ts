// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPURegionMask's per-frame region polygon (a lasso that the sliders scale and rotate).
export const parameters: GpuSceneParameter[] = [
  {id: 'size', label: 'region size', min: 0.08, max: 0.5, step: 0.01, default: 0.3},
  {id: 'rotation', label: 'region rotation', min: 0, max: 6.28, step: 0.1, default: 0.6},
  {id: 'notch', label: 'region notch', min: 0.2, max: 1, step: 0.05, default: 0.45}
];
