// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTextureShading({levelCount}) and getGPUTextureShadingParameterValues({detail, gain}).
export const parameters: GpuSceneParameter[] = [
  {id: 'detail', label: 'detail', min: 0, max: 2, step: 0.1, default: 0.5},
  {id: 'gain', label: 'gain', min: 0.5, max: 8, step: 0.5, default: 2},
  {id: 'levelCount', label: 'levelCount', min: 2, max: 8, step: 1, default: 6}
];
