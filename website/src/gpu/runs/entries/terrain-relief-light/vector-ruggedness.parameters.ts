// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUTerrainVectorRuggednessParameterValues({radius, zFactor}).
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 1, max: 7, step: 1, default: 3},
  {id: 'zFactor', label: 'zFactor', min: 0.5, max: 4, step: 0.25, default: 1}
];
