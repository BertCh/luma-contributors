// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `operation.type` and `operation.k` of GPUCellTopology plus the grid family; all fix the output
// stride, so they are compile-time. Cell color is the grid distance (disk) from the seed cell.
export const parameters: GpuSceneParameter[] = [
  {id: 'family', label: 'family', min: 0, max: 1, step: 1, default: 0, options: ['h3', 'quadbin']},
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['disk', 'ring']
  },
  {id: 'k', label: 'k', min: 1, max: 4, step: 1, default: 3}
];
