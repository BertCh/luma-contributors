// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'seed', label: 'seed', min: 1, max: 40, step: 1, default: 7},
  {
    id: 'priority',
    label: 'priority',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['largest-degree-first', 'random']
  }
];
