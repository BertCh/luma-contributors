// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'channel',
    label: 'Component shown',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['first (red)', 'second (green)', 'third (blue)']
  },
  {id: 'sigmaCount', label: 'sigmaCount', min: 0.5, max: 4, step: 0.25, default: 2}
];
