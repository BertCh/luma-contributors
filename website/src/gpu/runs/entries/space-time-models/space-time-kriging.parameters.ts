// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Per-frame inputs of GPUSpaceTimeKriging: the query time (the `queryTimes` view), and the
// temporal and spatial ranges of the covariance written by getGPUSpaceTimeKrigingParameterValues.
// Times are days (converted to the seconds of the temporal context), spatial range in map units.
export const parameters: GpuSceneParameter[] = [
  {id: 'queryTime', label: 'queryTime', min: -0.6, max: 1.4, step: 0.05, default: 0.5, unit: 'day'},
  {
    id: 'temporalRange',
    label: 'temporalRange',
    min: 0.1,
    max: 5,
    step: 0.05,
    default: 1,
    unit: 'day'
  },
  {id: 'spatialRange', label: 'spatialRange', min: 0.03, max: 1.5, step: 0.01, default: 0.3}
];
