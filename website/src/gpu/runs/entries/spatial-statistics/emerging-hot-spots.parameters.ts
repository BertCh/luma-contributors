// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame settings of GPUEmergingHotSpots (`radius`, `temporalWindow`), plus the share of observed temporal contrast.
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 1, max: 3, step: 1, default: 1, unit: ' cells'},
  {
    id: 'temporalWindow',
    label: 'temporalWindow',
    min: 0,
    max: 3,
    step: 1,
    default: 1,
    unit: ' slices'
  },
  {id: 'trend', label: 'Data: temporal contrast', min: 0, max: 1, step: 0.05, default: 1}
];
