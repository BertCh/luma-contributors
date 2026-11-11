// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'blockSize',
    label: 'Data: zone size',
    min: 0,
    max: 3,
    step: 1,
    default: 2,
    options: ['2 px', '4 px', '6 px', '9 px']
  },
  {
    id: 'shown',
    label: 'Output shown',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['meanLengths', 'means (NIR component)']
  }
];
