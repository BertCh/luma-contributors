// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialWeightsAlgebra's `operation`; k sets the left operand, radius the right one.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 3,
    step: 1,
    default: 1,
    options: ['union', 'intersection', 'difference', 'symmetricDifference']
  },
  {id: 'k', label: 'Data: left k', min: 2, max: 6, step: 1, default: 3},
  {
    id: 'radius',
    label: 'Data: right radius',
    min: 0.06,
    max: 0.16,
    step: 0.01,
    default: 0.1
  }
];
