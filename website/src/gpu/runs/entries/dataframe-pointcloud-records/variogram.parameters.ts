// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUVariogram's compile-time `lagCount`, its per-frame `maximumDistance`, and the nugget noise.
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
  {id: 'noise', label: 'Data: nugget noise', min: 0, max: 2, step: 0.1, default: 0}
];
