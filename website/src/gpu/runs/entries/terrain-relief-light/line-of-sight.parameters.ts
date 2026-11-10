// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to getGPUTerrainLineOfSightParameterValues({observerHeight, targetHeight}); the observer is the first element of every pair.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'observerColumn',
    label: 'pairs: observer column',
    min: 0,
    max: 100,
    step: 1,
    default: 35,
    unit: '%'
  },
  {
    id: 'observerRow',
    label: 'pairs: observer row',
    min: 0,
    max: 100,
    step: 1,
    default: 55,
    unit: '%'
  },
  {id: 'observerHeight', label: 'observerHeight', min: 0, max: 80, step: 1, default: 6, unit: ' m'},
  {id: 'targetHeight', label: 'targetHeight', min: 0, max: 40, step: 1, default: 2, unit: ' m'}
];

export const legend: RunLegendItem[] = [
  {label: 'target clear (clearance above terrain)', color: '#c4242b'},
  {label: 'target blocked (negative clearance)', color: '#1f4fc4'}
];
