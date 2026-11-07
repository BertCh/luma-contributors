// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'markerCount', label: 'Data: markers', min: 2, max: 40, step: 1, default: 14},
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['fifo', 'minimum-spanning-forest']
  },
  {
    id: 'connectivity',
    label: 'connectivity',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['4', '8']
  }
];
