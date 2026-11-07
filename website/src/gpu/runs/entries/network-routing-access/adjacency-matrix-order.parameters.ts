// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// tieKeys is an optional compile-time input of GPUAdjacencyMatrixOrder.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'tieKeys',
    label: 'tieKeys',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['none', 'degree']
  },
  {id: 'communities', label: 'Data: communities', min: 2, max: 8, step: 1, default: 5}
];
