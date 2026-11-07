// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialScanStatistic's `maximumWindowZones` and its per-frame `parameters`
// (rate direction, maximum population fraction), plus the planted excess rate.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumWindowZones', label: 'maximumWindowZones', min: 4, max: 36, step: 4, default: 16},
  {
    id: 'maximumPopulationFraction',
    label: 'maximumPopulationFraction',
    min: 0.05,
    max: 0.5,
    step: 0.05,
    default: 0.25
  },
  {
    id: 'rateDirection',
    label: 'rateDirection',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['both', 'high', 'low']
  },
  {id: 'excessRate', label: 'Data: excessRate', min: 0.05, max: 0.3, step: 0.01, default: 0.15}
];
