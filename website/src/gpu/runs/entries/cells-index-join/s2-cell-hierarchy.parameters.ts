// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `operation` and `level` are GPUS2CellHierarchy props (compile time); `inputLevel` is the level of
// the seed cells that are keyed first.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['parent', 'children', 'neighbors']
  },
  {id: 'level', label: 'level', min: 6, max: 9, step: 1, default: 8},
  {id: 'inputLevel', label: 'Data: inputLevel', min: 9, max: 11, step: 1, default: 10}
];
