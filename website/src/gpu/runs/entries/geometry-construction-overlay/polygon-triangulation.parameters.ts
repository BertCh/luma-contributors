// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'irregularity', label: 'Data: irregularity', min: 0, max: 0.8, step: 0.02, default: 0.45},
  {
    id: 'hasHoles',
    label: 'Data: holes',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  }
];
