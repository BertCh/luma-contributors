// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to getGPUTerrainPeakSnapParameterValues({radius, maximumMove, maximumHeightChange}); the offset moves the candidates.
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 60, max: 900, step: 30, default: 600, unit: ' m'},
  {id: 'maximumMove', label: 'maximumMove', min: 60, max: 900, step: 30, default: 600, unit: ' m'},
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
    default: 2,
    unit: ' cells'
  }
];

export const legend: RunLegendItem[] = [
  {label: 'snapped to the summit', color: '#33cc66'},
  {label: 'unchanged', color: '#8c94a6'},
  {label: 'on the ring (flank, kept)', color: '#f2a633'},
  {label: 'move too long', color: '#e64040'},
  {label: 'height off', color: '#a64dd9'}
];
