// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUClassificationFit's `classCount`; the period only chooses which map is shown.
export const parameters: GpuSceneParameter[] = [
  {id: 'classCount', label: 'classCount', min: 2, max: 8, step: 1, default: 5},
  {
    id: 'quantile',
    label: 'Data: breaks',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['equal interval', 'quantile']
  },
  {id: 'period', label: 'Show: period', min: 0, max: 5, step: 1, default: 0}
];
