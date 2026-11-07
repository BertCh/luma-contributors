// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUMultiScaleRelief({featureMaximum, scalingFactor}) and getGPUMultiScaleReliefParameterValues({verticalExaggeration}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'featureMaximum',
    label: 'featureMaximum',
    min: 8,
    max: 60,
    step: 4,
    default: 24,
    unit: ' cells'
  },
  {id: 'scalingFactor', label: 'scalingFactor', min: 1, max: 3, step: 1, default: 2},
  {
    id: 'verticalExaggeration',
    label: 'verticalExaggeration',
    min: 0.5,
    max: 4,
    step: 0.25,
    default: 1
  }
];
