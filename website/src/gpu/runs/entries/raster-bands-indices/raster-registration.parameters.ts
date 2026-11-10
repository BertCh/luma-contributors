// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'upsampleFactor', label: 'upsampleFactor', min: 1, max: 32, step: 1, default: 10},
  {
    id: 'shiftX',
    label: 'true shift x',
    min: 0.05,
    max: 0.95,
    step: 0.01,
    default: 0.62,
    unit: 'px'
  },
  {
    id: 'shiftY',
    label: 'true shift y',
    min: 0.05,
    max: 0.95,
    step: 0.01,
    default: 0.27,
    unit: 'px'
  }
];
