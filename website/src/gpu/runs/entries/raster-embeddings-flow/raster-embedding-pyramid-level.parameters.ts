// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'scale', label: 'scale', min: 0, max: 3, step: 1, default: 2, options: ['1', '2', '4', '8']},
  {
    id: 'shown',
    label: 'Output shown',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['vectors (NIR component)', 'meanLength']
  },
  {id: 'noData', label: 'Data: invalid pixels', min: 0, max: 0.6, step: 0.05, default: 0.1}
];
