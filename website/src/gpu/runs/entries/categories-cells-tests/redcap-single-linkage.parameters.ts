// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `regionCount` is the per-frame target of the spanning-forest cut. `standardize` is a compile-time
// prop: whether each attribute column is scaled to mean 0 and unit deviation first.
export const parameters: GpuSceneParameter[] = [
  {id: 'regionCount', label: 'regionCount', min: 2, max: 30, step: 1, default: 8},
  {
    id: 'standardize',
    label: 'standardize',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  }
];
