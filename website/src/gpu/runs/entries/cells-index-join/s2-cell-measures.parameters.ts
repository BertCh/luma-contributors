// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `measure` picks the column of GPUS2CellMeasures' `measures` output (area or perimeter) that is
// drawn; `level` is the S2 level of the cells being measured.
export const parameters: GpuSceneParameter[] = [
  {id: 'level', label: 'Data: level', min: 6, max: 9, step: 1, default: 8},
  {
    id: 'measure',
    label: 'measure',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['area', 'perimeter']
  }
];
