// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// firstDeparture and step are the two rows of GPUNetworkTravelTimePercentiles' per-frame `window`.
// departureCount is compile-time. percentile picks which of the computed percentiles is drawn.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'firstDeparture',
    label: 'window: first departure',
    min: 0,
    max: 1.5,
    step: 0.05,
    default: 0.4
  },
  {id: 'step', label: 'window: step', min: 0, max: 0.2, step: 0.01, default: 0.06},
  {id: 'departureCount', label: 'departureCount', min: 2, max: 24, step: 1, default: 12},
  {
    id: 'percentile',
    label: 'percentile drawn',
    min: 0,
    max: 4,
    step: 1,
    default: 4,
    options: ['p5', 'p50', 'p80', 'p95', 'p95 - p50']
  }
];
