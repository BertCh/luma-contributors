// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUWeightedOverlayParameterValues({layers: [{weight}], normalizeWeights}).
export const parameters: GpuSceneParameter[] = [
  {id: 'greenWeight', label: 'layers[0].weight (water)', min: -1, max: 1, step: 0.05, default: 0.5},
  {
    id: 'elevationWeight',
    label: 'layers[1].weight (elevation)',
    min: -1,
    max: 1,
    step: 0.05,
    default: -0.5
  },
  {
    id: 'nirWeight',
    label: 'layers[2].weight (vegetation)',
    min: -1,
    max: 1,
    step: 0.05,
    default: 0.8
  },
  {
    id: 'normalizeWeights',
    label: 'normalizeWeights',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  }
];
