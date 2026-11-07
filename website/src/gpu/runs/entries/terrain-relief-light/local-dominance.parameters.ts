// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPULocalDominance({minimumRadius, maximumRadius}) and getGPULocalDominanceParameterValues({observerHeight}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'observerHeight',
    label: 'observerHeight',
    min: 0.5,
    max: 40,
    step: 0.5,
    default: 8,
    unit: ' m'
  },
  {
    id: 'minimumRadius',
    label: 'minimumRadius',
    min: 2,
    max: 10,
    step: 1,
    default: 4,
    unit: ' cells'
  },
  {
    id: 'maximumRadius',
    label: 'maximumRadius',
    min: 12,
    max: 30,
    step: 2,
    default: 20,
    unit: ' cells'
  }
];
