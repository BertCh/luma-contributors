// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Which of the requested `columns[].statistics` is drawn, the `mask` of rows kept, and data noise.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'statistic',
    label: 'statistics',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['mean', 'median', 'standardDeviation', 'maximum']
  },
  {id: 'mask', label: 'mask', min: 10, max: 100, step: 5, default: 70, unit: '% of rows'},
  {id: 'noise', label: 'Data: noise', min: 0.1, max: 1.5, step: 0.05, default: 0.5}
];
