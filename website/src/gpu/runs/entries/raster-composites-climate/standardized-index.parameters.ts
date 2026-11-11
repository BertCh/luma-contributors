// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'month', label: 'month evaluated', min: 0, max: 59, step: 1, default: 56, unit: ' of 60'},
  {
    id: 'distribution',
    label: 'distribution',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['gamma', 'fisk']
  }
];
