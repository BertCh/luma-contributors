// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `family` and `resolution` are GPUCellAggregation props (compile time). `measure` picks the output
// column that is drawn: sumValues, minimums or maximums.
export const parameters: GpuSceneParameter[] = [
  {id: 'family', label: 'family', min: 0, max: 1, step: 1, default: 0, options: ['h3', 'quadbin']},
  {id: 'resolution', label: 'resolution', min: 3, max: 6, step: 1, default: 5},
  {
    id: 'measure',
    label: 'measure',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['sumValues', 'minimums', 'maximums']
  }
];
