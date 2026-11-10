// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'height', label: 'marker offset', min: 0, max: 0.6, step: 0.02, default: 0.25},
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['dilation', 'erosion']
  },
  {id: 'maximumIterations', label: 'maximumIterations', min: 4, max: 160, step: 4, default: 120}
];
