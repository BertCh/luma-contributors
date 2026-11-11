// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'activeSliceCount',
    label: 'activeSliceCount',
    min: 24,
    max: 288,
    step: 12,
    default: 288,
    unit: ' months'
  },
  {
    id: 'distribution',
    label: 'distribution',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['gamma', 'fisk']
  },
  {
    id: 'month',
    label: 'month shown (slot)',
    min: 0,
    max: 11,
    step: 1,
    default: 8,
    options: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  }
];
