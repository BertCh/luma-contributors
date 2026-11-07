// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// All three are GPUEdgeBundling's per-frame `parameters` words.
export const parameters: GpuSceneParameter[] = [
  {id: 'activeIterations', label: 'activeIterations', min: 0, max: 15, step: 1, default: 12},
  {id: 'kernelRadius', label: 'kernelRadius', min: 0.01, max: 0.12, step: 0.005, default: 0.05},
  {id: 'smoothing', label: 'smoothing', min: 0, max: 1, step: 0.05, default: 0.5}
];
