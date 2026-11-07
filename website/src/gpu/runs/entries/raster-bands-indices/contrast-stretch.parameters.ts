// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPURasterStretchParameterValues({mode, percentiles, gamma}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'mode',
    label: 'mode',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['linear', 'percentile', 'equalize']
  },
  {id: 'percentileLow', label: 'percentiles[0]', min: 0, max: 30, step: 1, default: 2},
  {id: 'percentileHigh', label: 'percentiles[1]', min: 70, max: 100, step: 1, default: 98},
  {id: 'gamma', label: 'gamma', min: 0.3, max: 3, step: 0.1, default: 1}
];
