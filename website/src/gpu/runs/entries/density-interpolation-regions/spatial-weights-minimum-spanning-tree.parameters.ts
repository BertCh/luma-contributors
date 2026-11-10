// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialWeightsMinimumSpanningTree's `columnCount` and `standardize`.
export const parameters: GpuSceneParameter[] = [
  {id: 'columnCount', label: 'columnCount', min: 1, max: 2, step: 1, default: 2},
  {
    id: 'standardize',
    label: 'standardize',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  },
  {id: 'tractRows', label: 'Data: tract window size', min: 5, max: 10, step: 1, default: 7}
];
