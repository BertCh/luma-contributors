// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPURasterConditionalParameterValues({comparison, threshold, upperThreshold}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'comparison',
    label: 'comparison',
    min: 0,
    max: 4,
    step: 1,
    default: 4,
    options: ['<', '<=', '>', '>=', 'between']
  },
  {id: 'threshold', label: 'threshold', min: 0.1, max: 0.9, step: 0.01, default: 0.45},
  {id: 'upperThreshold', label: 'upperThreshold', min: 0.1, max: 1, step: 0.01, default: 0.7}
];
