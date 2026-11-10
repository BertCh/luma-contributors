// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to GPUPointHorizonVisibility({observers, targets}) and getGPUPointHorizonVisibilityParameterValues({maximumDistance}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'observerColumn',
    label: 'observers: column',
    min: 0,
    max: 100,
    step: 1,
    default: 25,
    unit: '%'
  },
  {id: 'observerRow', label: 'observers: row', min: 0, max: 100, step: 1, default: 60, unit: '%'},
  {
    id: 'observerHeight',
    label: 'observers: height',
    min: 0,
    max: 300,
    step: 5,
    default: 20,
    unit: ' m'
  },
  {id: 'targetHeight', label: 'targets: height', min: 0, max: 150, step: 5, default: 5, unit: ' m'}
];

export const legend: RunLegendItem[] = [
  {label: 'summit visible', color: '#33d966'},
  {label: 'summit hidden by the horizon', color: '#451a66'}
];
