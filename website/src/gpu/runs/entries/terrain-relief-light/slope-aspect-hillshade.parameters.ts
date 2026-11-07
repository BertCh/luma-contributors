// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to getGPUTerrainDerivativesParameterValues({zFactor, azimuthDegrees, altitudeDegrees}).
// `view` only chooses which of the three outputs is drawn.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'view',
    label: 'output',
    min: 0,
    max: 2,
    step: 1,
    default: 2,
    options: ['slope', 'aspect', 'hillshade']
  },
  {
    id: 'azimuthDegrees',
    label: 'azimuthDegrees',
    min: 0,
    max: 360,
    step: 5,
    default: 315,
    unit: '°'
  },
  {
    id: 'altitudeDegrees',
    label: 'altitudeDegrees',
    min: 5,
    max: 85,
    step: 5,
    default: 45,
    unit: '°'
  },
  {id: 'zFactor', label: 'zFactor', min: 0.5, max: 8, step: 0.5, default: 3}
];

export const legend: RunLegendItem[] = [
  {label: 'slope: flat (pale) to steep (dark)', color: '#a8204e'},
  {label: 'aspect: hue wheel, 0° north clockwise', color: '#3aa0d8'},
  {label: 'hillshade: shadow to lit', color: '#999999'}
];
