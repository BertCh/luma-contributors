// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// operation and selectCount are compile-time props; capacity scales the capacities view each frame.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['p-center', 'capacitated-p-median']
  },
  {id: 'selectCount', label: 'selectCount', min: 2, max: 6, step: 1, default: 4},
  {id: 'capacity', label: 'capacities', min: 0.3, max: 3, step: 0.1, default: 1.2}
];
