// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'statistic',
    label: 'Output shown',
    min: 0,
    max: 5,
    step: 1,
    default: 5,
    options: [
      'difference',
      'percentChange',
      'tStatistic',
      'senSlope',
      'mannKendallZ',
      'significance'
    ]
  },
  {id: 'splitSlice', label: 'splitSlice', min: 3, max: 9, step: 1, default: 6},
  {id: 'alpha', label: 'alpha', min: 0.001, max: 0.3, step: 0.001, default: 0.05},
  {id: 'magnitude', label: 'Data: change', min: 0, max: 0.4, step: 0.02, default: 0.2}
];
