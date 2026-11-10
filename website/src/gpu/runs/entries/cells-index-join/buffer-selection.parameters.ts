// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `distance` is GPUBufferSelection's per-frame view (no recompile). `bend` reshapes the river, a
// per-frame rewrite of the segment features.
export const parameters: GpuSceneParameter[] = [
  {id: 'distance', label: 'distance', min: 0.01, max: 0.25, step: 0.005, default: 0.08},
  {id: 'bend', label: 'Data: bend', min: 0, max: 0.35, step: 0.01, default: 0.18}
];
