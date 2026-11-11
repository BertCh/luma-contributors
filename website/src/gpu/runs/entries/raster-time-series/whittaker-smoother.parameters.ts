// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'smoothingParameter',
    label: 'smoothingParameter',
    min: 0,
    max: 6,
    step: 1,
    default: 2,
    options: ['0.5', '2', '6', '20', '60', '200', '1000']
  },
  {id: 'differenceOrder', label: 'differenceOrder', min: 1, max: 3, step: 1, default: 2},
  {
    id: 'reweightFunction',
    label: 'reweightFunction',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['bisquare', 'upper-envelope']
  },
  {id: 'cloudFraction', label: 'Data: cloud fraction', min: 0, max: 0.4, step: 0.02, default: 0.18}
];
