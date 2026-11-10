// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPURipley's per-frame `parameters` (maximumDistance, edgeCorrection), its compile-time
// `radiusCount`, and the clustering of the input pattern.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 0.05,
    max: 0.3,
    step: 0.01,
    default: 0.2
  },
  {
    id: 'edgeCorrection',
    label: 'edgeCorrection',
    min: 0,
    max: 2,
    step: 1,
    default: 2,
    options: ['none', 'border', 'isotropic']
  },
  {id: 'radiusCount', label: 'radiusCount', min: 8, max: 48, step: 4, default: 32},
  {id: 'clustering', label: 'Data: observed flash share', min: 0, max: 1, step: 0.05, default: 1}
];
