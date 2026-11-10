// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'distance', label: 'distance', min: 0.005, max: 0.1, step: 0.0025, default: 0.04},
  {
    id: 'geometryType',
    label: 'geometryType',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['points', 'lines', 'polygons']
  }
];
