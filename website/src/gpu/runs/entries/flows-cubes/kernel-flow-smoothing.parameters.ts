// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// neighborhoodSize and bandwidthRatio are the two per-frame settings of GPUKernelFlowSmoothing
// (locationWeights are ones, so the neighborhood is that many tracts).
export const parameters: GpuSceneParameter[] = [
  {id: 'neighborhoodSize', label: 'neighborhoodSize', min: 0, max: 40, step: 1, default: 12},
  {id: 'bandwidthRatio', label: 'bandwidthRatio', min: 0.15, max: 1.5, step: 0.05, default: 0.5}
];
