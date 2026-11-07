// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointBuffer's per-frame radius in `parameters` and its compile-time `verticesPerRing`.
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 0.01, max: 0.2, step: 0.005, default: 0.07},
  {id: 'verticesPerRing', label: 'verticesPerRing', min: 3, max: 32, step: 1, default: 16},
  {id: 'pointCount', label: 'Data: points', min: 3, max: 24, step: 1, default: 10}
];
