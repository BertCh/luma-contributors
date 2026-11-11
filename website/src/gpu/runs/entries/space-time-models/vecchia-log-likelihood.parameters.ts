// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The GPUVecchiaCovariance written by getGPUVecchiaLogLikelihoodParameterValues: per frame, so the
// pad moves through (range, sill) without a recompile. Ranges are in map units (picture height 1).
export const parameters: GpuSceneParameter[] = [
  {id: 'range', label: 'range', min: 0.01, max: 6, step: 0.005, default: 0.5},
  {id: 'sill', label: 'sill', min: 0.1, max: 20, step: 0.05, default: 1},
  {id: 'nugget', label: 'nugget', min: 0, max: 1, step: 0.05, default: 0.2},
  {
    id: 'model',
    label: 'model',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['exponential', 'matern-3/2']
  }
];
