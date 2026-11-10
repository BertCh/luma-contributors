// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame `epsilon` of GPUInequality, which output is drawn, and the income spread.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 3,
    step: 1,
    default: 2,
    options: ['gini', 'theilT', 'atkinson', 'hoover']
  },
  {id: 'epsilon', label: 'epsilon', min: 0.1, max: 4, step: 0.1, default: 1}
];
