// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// resolution, measure and classify are compile-time props of addPeriodComparisonRecipe; the shift
// moves the second period's cluster.
export const parameters: GpuSceneParameter[] = [
  {id: 'resolution', label: 'resolution', min: 11, max: 13, step: 1, default: 12},
  {
    id: 'classify',
    label: 'classify',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['delta', 'zScore', 'percentChange']
  },
  {id: 'shift', label: 'Data: clusterShift', min: 0, max: 1, step: 0.05, default: 0.6}
];
