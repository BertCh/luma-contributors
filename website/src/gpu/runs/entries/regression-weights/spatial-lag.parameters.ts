// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialLag's `normalize` prop, plus the data noise of the input values.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'normalize',
    label: 'normalize',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  },
  {
    id: 'noise',
    label: 'Data: noise',
    min: 0,
    max: 1.5,
    step: 0.05,
    default: 0.8
  }
];
