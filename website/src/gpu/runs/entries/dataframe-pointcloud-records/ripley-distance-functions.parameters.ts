// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPURipleyDistanceFunctions' per-frame `parameters` (maximumDistance, edgeCorrection),
// its compile-time `radiusCount`, and the clustering of the input pattern.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 0.02,
    max: 0.15,
    step: 0.01,
    default: 0.08
  },
  {
    id: 'edgeCorrection',
    label: 'edgeCorrection',
    min: 0,
    max: 3,
    step: 1,
    default: 1,
    options: ['none', 'border', 'kaplan-meier', 'hanisch']
  },
  {id: 'radiusCount', label: 'radiusCount', min: 8, max: 40, step: 4, default: 24},
  {id: 'clustering', label: 'Data: observed flash share', min: 0, max: 1, step: 0.05, default: 1}
];
