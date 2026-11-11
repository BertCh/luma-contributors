// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'timeStep', label: 'timeStep', min: -0.03, max: 0.03, step: 0.0025, default: 0.015},
  {
    id: 'stepCount',
    label: 'stepCount',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['16', '48', '96']
  }
];
