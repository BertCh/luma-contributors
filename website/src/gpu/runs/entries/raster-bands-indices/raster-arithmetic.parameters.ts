// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPURasterArithmeticParameterValues({operation, scaleB, clampMaximum}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 4,
    step: 1,
    default: 0,
    options: ['normalizedDifference', 'subtract', 'multiply', 'maximum', 'absoluteDifference']
  },
  {id: 'scaleB', label: 'scaleB', min: 0.2, max: 2, step: 0.05, default: 1},
  {id: 'clampMaximum', label: 'clampMaximum', min: 0.1, max: 1, step: 0.05, default: 1}
];
