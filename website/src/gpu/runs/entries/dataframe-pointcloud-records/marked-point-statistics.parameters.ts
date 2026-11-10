// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUMarkedPointStatistics' `typePair`, its compile-time `radiusCount`, and the attraction
// of type 1 points to type 0 points in the input.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'typePair',
    label: 'typePair',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['[0, 1]', '[0, 0]', '[1, 1]']
  },
  {id: 'radiusCount', label: 'radiusCount', min: 8, max: 40, step: 4, default: 24},
  {
    id: 'areaThreshold',
    label: 'Data: flash area cutoff',
    min: 80,
    max: 600,
    step: 20,
    default: 200,
    unit: 'km²'
  }
];
