// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'statistic',
    label: 'Statistic shown',
    min: 0,
    max: 5,
    step: 1,
    default: 0,
    options: ['means', 'maximums', 'minimums', 'sums', 'standardDeviations', 'valueCounts']
  },
  {id: 'gridWidth', label: 'grid.width', min: 16, max: 120, step: 4, default: 48, unit: 'cells'}
];
