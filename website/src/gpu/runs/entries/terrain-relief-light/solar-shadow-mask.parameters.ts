// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUSolarShadowMaskParameterValues({azimuthDegrees, altitudeDegrees, ambientIntensity}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'azimuthDegrees',
    label: 'azimuthDegrees',
    min: 0,
    max: 360,
    step: 5,
    default: 135,
    unit: '°'
  },
  {
    id: 'altitudeDegrees',
    label: 'altitudeDegrees',
    min: 3,
    max: 80,
    step: 1,
    default: 12,
    unit: '°'
  },
  {id: 'ambientIntensity', label: 'ambientIntensity', min: 0, max: 0.6, step: 0.05, default: 0.2}
];
