// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['mean', 'minimum', 'maximum', 'median']
  },
  {
    id: 'cloudCover',
    label: 'Data: missing observations',
    min: 0,
    max: 0.8,
    step: 0.05,
    default: 0.4
  }
];
