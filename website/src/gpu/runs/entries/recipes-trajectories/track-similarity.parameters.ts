// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The measure chooses which per-pair output GPUTrackSimilarity writes (compile-time); the noise
// slider changes the track vertices (per-frame input).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'measure',
    label: 'measure',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['hausdorff', 'frechet', 'dynamicTimeWarping']
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 0.3, step: 0.01, default: 0.08}
];
