// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainHorizon({directionCount, maximumRadius}) and getGPUTerrainHorizonParameterValues({zFactor}).
export const parameters: GpuSceneParameter[] = [
  {id: 'directionCount', label: 'directionCount', min: 8, max: 32, step: 8, default: 16},
  {
    id: 'maximumRadius',
    label: 'maximumRadius',
    min: 10,
    max: 60,
    step: 10,
    default: 40,
    unit: ' cells'
  },
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['skyViewFactor', 'positiveOpenness', 'negativeOpenness']
  },
  {id: 'zFactor', label: 'zFactor', min: 0.5, max: 4, step: 0.25, default: 1}
];
