// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'shape',
    label: 'shape',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['circle', 'sector', 'ellipse']
  },
  {id: 'segmentCount', label: 'segmentCount', min: 3, max: 48, step: 1, default: 12},
  {id: 'radiusScale', label: 'radiusScale', min: 0.3, max: 1.4, step: 0.05, default: 1}
];
