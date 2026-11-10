// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'statistic',
    label: 'Statistic shown',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['means', 'totals', 'coveredArea', 'validArea']
  },
  {id: 'gridWidth', label: 'grid.width', min: 6, max: 80, step: 2, default: 16, unit: 'cells'}
];
