// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to GPUGeomorphons({searchRadius, comparison}) and getGPUGeomorphonsParameterValues({flatThresholdDegrees}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'searchRadius',
    label: 'searchRadius',
    min: 4,
    max: 24,
    step: 1,
    default: 12,
    unit: ' cells'
  },
  {
    id: 'flatThresholdDegrees',
    label: 'flatThresholdDegrees',
    min: 0.5,
    max: 10,
    step: 0.5,
    default: 3,
    unit: '°'
  },
  {
    id: 'comparison',
    label: 'comparison',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['anglev1', 'anglev2', 'anglev2-distance']
  }
];

export const legend: RunLegendItem[] = [
  {label: 'flat', color: '#cccbc7'},
  {label: 'peak', color: '#bf1a1a'},
  {label: 'ridge', color: '#f26626'},
  {label: 'shoulder', color: '#fab333'},
  {label: 'spur', color: '#b3cc40'},
  {label: 'slope', color: '#f2edb3'},
  {label: 'hollow', color: '#8ccce6'},
  {label: 'footslope', color: '#33a699'},
  {label: 'valley', color: '#2659cc'},
  {label: 'pit', color: '#1a1a59'}
];
