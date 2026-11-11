// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'iterationCount',
    label: 'iterationCount',
    min: 0,
    max: 5,
    step: 1,
    default: 4,
    options: ['1', '2', '4', '8', '16', '32']
  },
  {
    id: 'distanceEpsilon',
    label: 'distanceEpsilon',
    min: 0,
    max: 5,
    step: 1,
    default: 0,
    options: ['1e-5', '1e-3', '0.01', '0.05', '0.2', '0.5']
  },
  {id: 'haze', label: 'Data: undetected haze', min: 0, max: 1, step: 0.05, default: 1}
];
