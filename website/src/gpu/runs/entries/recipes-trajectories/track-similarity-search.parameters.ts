// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// threshold is the per-frame threshold buffer of GPUTrackSimilaritySearch; metric is compile-time.
export const parameters: GpuSceneParameter[] = [
  {id: 'threshold', label: 'threshold', min: 0.05, max: 1.5, step: 0.05, default: 0.4},
  {
    id: 'metric',
    label: 'metric',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['frechet', 'dynamic-time-warping']
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 0.3, step: 0.01, default: 0.08}
];
