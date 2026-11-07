// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'shown',
    label: 'Output shown',
    min: 0,
    max: 3,
    step: 1,
    default: 3,
    options: ['water fraction', 'soil fraction', 'vegetation fraction', 'residual']
  },
  {
    id: 'constraint',
    label: 'constraint',
    min: 0,
    max: 2,
    step: 1,
    default: 2,
    options: ['unconstrained', 'nonnegative', 'simplex']
  },
  {id: 'unknown', label: 'Data: unknown material', min: 0, max: 1, step: 0.05, default: 0.5},
  {id: 'noise', label: 'Data: noise', min: 0, max: 0.1, step: 0.005, default: 0.03}
];
