// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUFocalStatistics' per-frame `radius` and `shape`; the statistic picks which of the
// operation's output buffers is drawn.
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 1, max: 8, step: 1, default: 3, unit: ' cells'},
  {id: 'shape', label: 'shape', min: 0, max: 1, step: 1, default: 1, options: ['square', 'circle']},
  {
    id: 'statistic',
    label: 'output',
    min: 0,
    max: 4,
    step: 1,
    default: 0,
    options: ['mean', 'min', 'max', 'range', 'standardDeviation']
  }
];
