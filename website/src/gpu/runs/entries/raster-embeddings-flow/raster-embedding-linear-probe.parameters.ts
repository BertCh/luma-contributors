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
    default: 0,
    options: ['labels', 'probability of water', 'probability of forest', 'probability of open']
  },
  {id: 'weightScale', label: 'Data: weights scale', min: 0.1, max: 3, step: 0.1, default: 1}
];
