// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUVariogramModelFit's compile-time `weighting`, the variogram's maximumDistance that
// sets the empirical bins, and the nugget noise of the data.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'weighting',
    label: 'weighting',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['cressie', 'pairs', 'none']
  },
  {
    id: 'maximumDistance',
    label: 'Variogram: maximumDistance',
    min: 0.2,
    max: 0.7,
    step: 0.05,
    default: 0.5
  },
  {id: 'noise', label: 'Data: nugget noise', min: 0, max: 2, step: 0.1, default: 0.8}
];
