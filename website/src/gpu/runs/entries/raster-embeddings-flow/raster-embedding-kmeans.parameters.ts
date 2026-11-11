// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'clusterCount', label: 'clusterCount', min: 2, max: 8, step: 1, default: 5},
  {
    id: 'maximumIterations',
    label: 'maximumIterations',
    min: 0,
    max: 3,
    step: 1,
    default: 3,
    options: ['1', '2', '4', '16']
  }
];
