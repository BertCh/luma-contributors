// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to GPUTerrainWeissLandforms({smallScale, largeScale, standardization}) and getGPUTerrainWeissLandformsParameterValues({slopeThresholdDegrees}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'smallScale',
    label: 'smallScale.radius',
    min: 1,
    max: 6,
    step: 1,
    default: 2,
    unit: ' cells'
  },
  {
    id: 'largeScale',
    label: 'largeScale.radius',
    min: 8,
    max: 24,
    step: 1,
    default: 14,
    unit: ' cells'
  },
  {
    id: 'slopeThresholdDegrees',
    label: 'slopeThresholdDegrees',
    min: 1,
    max: 20,
    step: 1,
    default: 6,
    unit: '°'
  },
  {
    id: 'standardization',
    label: 'standardization',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['global', 'local']
  }
];

export const legend: RunLegendItem[] = [
  {label: 'canyon', color: '#1a1a59'},
  {label: 'midslope drainage', color: '#2659cc'},
  {label: 'upland drainage', color: '#8ccce6'},
  {label: 'u-shaped valley', color: '#33a699'},
  {label: 'plain', color: '#cccbc7'},
  {label: 'open slope', color: '#f2edb3'},
  {label: 'upper slope', color: '#fab333'},
  {label: 'local ridge', color: '#f26626'},
  {label: 'midslope ridge', color: '#cc3326'},
  {label: 'mountain top', color: '#8c0d1a'}
];
