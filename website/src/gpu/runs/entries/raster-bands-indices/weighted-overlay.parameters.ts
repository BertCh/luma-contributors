// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUWeightedOverlayParameterValues({layers: [{weight}], normalizeWeights}).
export const parameters: GpuSceneParameter[] = [
  {id: 'greenWeight', label: 'layers[0].weight (green)', min: -1, max: 1, step: 0.05, default: 0.5},
  {
    id: 'shortwaveWeight',
    label: 'layers[1].weight (SWIR)',
    min: -1,
    max: 1,
    step: 0.05,
    default: -0.5
  },
  {
    id: 'nirWeight',
    label: 'layers[2].weight (NIR)',
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
