// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Which of the requested `columns[].statistics` is drawn, the `mask`: the share of the map whose rows are kept.
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
  {id: 'mask', label: 'mask', min: 10, max: 100, step: 5, default: 70, unit: '% of map'}
];
