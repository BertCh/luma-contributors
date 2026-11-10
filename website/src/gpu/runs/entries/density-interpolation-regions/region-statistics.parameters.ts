// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPURegionStatistics' region (a per-frame view, so dragging never recompiles) and its
// histogram `binCount`.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'kind',
    label: 'region.kind',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['radius', 'rectangle']
  },
  {id: 'size', label: 'region size', min: 0.05, max: 0.5, step: 0.01, default: 0.22},
  {id: 'centerX', label: 'region center x', min: 0.1, max: 0.9, step: 0.01, default: 0.45},
  {id: 'centerY', label: 'region center y', min: 0.1, max: 0.9, step: 0.01, default: 0.5},
  {id: 'binCount', label: 'histogram.binCount', min: 4, max: 24, step: 2, default: 12}
];
