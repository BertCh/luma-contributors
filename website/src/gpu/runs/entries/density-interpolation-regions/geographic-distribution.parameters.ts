// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUGeographicDistribution's per-frame parameters; `spread` varies the data.
export const parameters: GpuSceneParameter[] = [
  {id: 'standardDeviations', label: 'standardDeviations', min: 0.5, max: 3, step: 0.5, default: 1},
  {
    id: 'ellipseConvention',
    label: 'ellipseConvention',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['arcgis', 'standard']
  },
  {id: 'spread', label: 'Data: spread', min: 0.5, max: 2, step: 0.1, default: 1}
];
