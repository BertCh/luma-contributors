// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'resultCount',
    label: 'resultCount',
    min: 0,
    max: 3,
    step: 1,
    default: 2,
    options: ['8', '32', '128', '512']
  },
  {
    id: 'context',
    label: 'Context shown',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['near infrared', 'vegetation index']
  }
];
