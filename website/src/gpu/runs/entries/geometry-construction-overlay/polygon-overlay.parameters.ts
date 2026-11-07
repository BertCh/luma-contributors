// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['intersection', 'union', 'difference', 'symmetric-difference']
  },
  {id: 'shift', label: 'Data: shift', min: 0, max: 1, step: 0.02, default: 0.4, unit: 'cells'}
];
