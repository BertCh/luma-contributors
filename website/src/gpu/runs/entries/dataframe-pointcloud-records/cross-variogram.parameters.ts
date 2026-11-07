// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUCrossVariogram's compile-time `lagCount`, the per-frame maximum distance in
// `parameters`, and the correlation between the two variables.
export const parameters: GpuSceneParameter[] = [
  {id: 'lagCount', label: 'lagCount', min: 6, max: 32, step: 2, default: 16},
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 0.1,
    max: 0.7,
    step: 0.05,
    default: 0.4
  },
  {id: 'correlation', label: 'Data: correlation', min: -1, max: 1, step: 0.1, default: 0.8}
];
