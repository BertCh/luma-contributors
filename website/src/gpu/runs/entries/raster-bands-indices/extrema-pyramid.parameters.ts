// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'firstBlockSize',
    label: 'firstBlockSize',
    min: 0,
    max: 5,
    step: 1,
    default: 2,
    options: ['1', '2', '4', '8', '16', '32']
  },
  {
    id: 'footprint',
    label: 'footprint',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['cell', 'bilinear']
  },
  {
    id: 'statistic',
    label: 'Level 0 shows',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['maximum', 'minimum']
  }
];
