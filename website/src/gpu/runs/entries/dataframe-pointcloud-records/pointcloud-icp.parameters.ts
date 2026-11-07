// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudICP's `iterations` and `maximumDistance` (compile-time), and the rotation
// that misaligns the source cloud.
export const parameters: GpuSceneParameter[] = [
  {id: 'iterations', label: 'iterations', min: 1, max: 20, step: 1, default: 12},
  {id: 'maximumDistance', label: 'maximumDistance', min: 0.03, max: 0.4, step: 0.01, default: 0.2},
  {id: 'rotation', label: 'Data: misalignment', min: 0, max: 20, step: 1, default: 8, unit: 'deg'}
];
