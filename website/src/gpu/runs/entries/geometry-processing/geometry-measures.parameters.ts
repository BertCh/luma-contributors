// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The measured quantity is a column choice; irregularity and holes change the input geometry.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'column',
    label: 'column',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['areas', 'lengths']
  },
  {id: 'irregularity', label: 'Data: irregularity', min: 0, max: 0.8, step: 0.05, default: 0.4}
];
