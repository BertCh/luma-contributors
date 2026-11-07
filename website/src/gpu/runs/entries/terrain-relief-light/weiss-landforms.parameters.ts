// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

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
