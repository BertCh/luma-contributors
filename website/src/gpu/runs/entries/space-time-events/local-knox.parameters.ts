// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Bound to the per-frame props of GPULocalKnox (`timeThreshold` and the permutation count in
// `parameters`) and the per-frame radius of the GPUNeighborSearch that lists the spatial pairs.
// The data control exchanges a share of the origin times between events.

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'timeThreshold',
    label: 'timeThreshold',
    min: 0.5,
    max: 20,
    step: 0.5,
    default: 3,
    unit: 'd'
  },
  {id: 'radius', label: 'radius', min: 0.25, max: 3, step: 0.125, default: 1, unit: 'km'},
  {id: 'observedShare', label: 'Data: observed times', min: 0, max: 1, step: 0.05, default: 1}
];
