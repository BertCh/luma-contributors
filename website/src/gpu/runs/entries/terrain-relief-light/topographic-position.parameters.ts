// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainTopographicPosition({scales, output}): three radii, small to large.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'smallRadius',
    label: 'scales[0].radius',
    min: 1,
    max: 4,
    step: 1,
    default: 2,
    unit: ' cells'
  },
  {
    id: 'largeRadius',
    label: 'scales[2].radius',
    min: 12,
    max: 32,
    step: 2,
    default: 24,
    unit: ' cells'
  },
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['maximumDeviation', 'maximumDeviationRadius']
  }
];
