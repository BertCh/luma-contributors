// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainCurvature({method, output.curvatures}) and getGPUTerrainCurvatureParameterValues({zFactor}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 2,
    step: 1,
    default: 2,
    options: ['evans-young', 'zevenbergen-thorne', 'florinsky']
  },
  {
    id: 'curvature',
    label: 'output.curvatures',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['profile', 'plan', 'mean', 'gaussian']
  },
  {id: 'zFactor', label: 'zFactor', min: 0.5, max: 4, step: 0.25, default: 1}
];
