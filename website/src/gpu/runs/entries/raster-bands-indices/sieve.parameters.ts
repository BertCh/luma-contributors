// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'minimumPixels', label: 'minimumPixels', min: 1, max: 120, step: 1, default: 30},
  {id: 'mode', label: 'mode', min: 0, max: 1, step: 1, default: 1, options: ['remove', 'merge']},
  {
    id: 'connectivity',
    label: 'connectivity',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['4', '8']
  }
];
