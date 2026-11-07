// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUReliefShadingParameterValues({lights, skyViewStrength, exposure, zFactor}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'azimuthDegrees',
    label: 'lights[0].azimuthDegrees',
    min: 0,
    max: 360,
    step: 5,
    default: 315,
    unit: '°'
  },
  {id: 'skyViewStrength', label: 'skyViewStrength', min: 0, max: 1, step: 0.05, default: 0.5},
  {id: 'exposure', label: 'exposure', min: 0.5, max: 2, step: 0.1, default: 1},
  {id: 'zFactor', label: 'zFactor', min: 0.5, max: 4, step: 0.25, default: 2}
];
