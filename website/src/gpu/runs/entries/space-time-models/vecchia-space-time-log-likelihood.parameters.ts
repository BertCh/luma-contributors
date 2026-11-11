// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The SpaceTimeVariogramModel written by getGPUVecchiaSpaceTimeLogLikelihoodParameterValues: per
// frame, so the pad moves through (spatialRange, temporalRange) without a recompile. Spatial range
// is in map units (picture height 1), temporal range in days, sill in squared log millimeters.
export const parameters: GpuSceneParameter[] = [
  {id: 'spatialRange', label: 'spatialRange', min: 0.03, max: 3, step: 0.005, default: 0.4},
  {
    id: 'temporalRange',
    label: 'temporalRange',
    min: 0.1,
    max: 10,
    step: 0.05,
    default: 0.5,
    unit: 'day'
  },
  {id: 'sill', label: 'sill', min: 0.1, max: 20, step: 0.05, default: 1},
  {id: 'nugget', label: 'nugget share', min: 0.01, max: 0.9, step: 0.01, default: 0.2}
];
