// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUVariableStandardization's `method` prop, plus the input skew.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 4,
    step: 1,
    default: 1,
    options: ['deviation-from-mean', 'z-score', 'mad', 'range-adjust', 'range-standardize']
  },
  {id: 'spread', label: 'Data: spread', min: 0.5, max: 1.8, step: 0.05, default: 1}
];
