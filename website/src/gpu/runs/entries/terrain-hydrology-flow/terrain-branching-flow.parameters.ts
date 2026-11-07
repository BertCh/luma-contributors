// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// mode is the operation's compile-time prop. flowExponent shapes the MFD fractions it consumes.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'mode',
    label: 'mode',
    min: 0,
    max: 4,
    step: 1,
    default: 0,
    options: [
      'transport',
      'expected-distance',
      'minimum-distance',
      'maximum-distance',
      'fractional-catchment'
    ]
  },
  {id: 'flowExponent', label: 'Data: flowExponent', min: 0.5, max: 6, step: 0.1, default: 1.1}
];
