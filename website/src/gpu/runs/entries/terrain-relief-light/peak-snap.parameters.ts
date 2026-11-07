// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUTerrainPeakSnapParameterValues({radius, maximumMove, maximumHeightChange}); the offset moves the candidates.
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 60, max: 600, step: 30, default: 300, unit: ' m'},
  {id: 'maximumMove', label: 'maximumMove', min: 60, max: 900, step: 30, default: 450, unit: ' m'},
  {
    id: 'maximumHeightChange',
    label: 'maximumHeightChange',
    min: 5,
    max: 200,
    step: 5,
    default: 60,
    unit: ' m'
  },
  {
    id: 'offset',
    label: 'Data: candidate offset',
    min: 0,
    max: 9,
    step: 1,
    default: 4,
    unit: ' cells'
  }
];
