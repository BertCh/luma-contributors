// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUTerrainCastShadowParameterValues({azimuthDegrees, altitudeDegrees, angularRadiusDegrees, zFactor}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'azimuthDegrees',
    label: 'azimuthDegrees',
    min: 0,
    max: 360,
    step: 5,
    default: 250,
    unit: '°'
  },
  {
    id: 'altitudeDegrees',
    label: 'altitudeDegrees',
    min: 3,
    max: 60,
    step: 1,
    default: 12,
    unit: '°'
  },
  {
    id: 'angularRadiusDegrees',
    label: 'angularRadiusDegrees',
    min: 0,
    max: 4,
    step: 0.25,
    default: 1,
    unit: '°'
  },
  {id: 'zFactor', label: 'zFactor', min: 0.5, max: 4, step: 0.25, default: 1}
];
