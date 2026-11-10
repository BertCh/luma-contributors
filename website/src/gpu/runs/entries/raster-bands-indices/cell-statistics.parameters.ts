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
    options: ['mean', 'maximum', 'range', 'standardDeviation', 'variety', 'counts']
  },
  {id: 'minimumValidCount', label: 'minimumValidCount', min: 1, max: 5, step: 1, default: 3},
  {id: 'cloudCover', label: 'Data: cloud cover', min: 0, max: 0.6, step: 0.05, default: 0.3}
];
