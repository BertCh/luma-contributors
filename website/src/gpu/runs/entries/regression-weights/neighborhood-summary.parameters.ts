// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `statistic` picks which column of the output table is drawn; includeFocal is a real prop.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'statistic',
    label: 'statistics column',
    min: 0,
    max: 7,
    step: 1,
    default: 3,
    options: ['count', 'weightSum', 'sum', 'mean', 'min', 'max', 'standardDeviation', 'median']
  },
  {
    id: 'includeFocal',
    label: 'includeFocal',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  },
  {id: 'k', label: 'Data: neighbors', min: 3, max: 10, step: 1, default: 6}
];
