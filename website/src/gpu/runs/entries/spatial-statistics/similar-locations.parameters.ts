// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSimilarLocations's `standardization` and the attribute `weights` of its per-frame
// `parameters`, plus which tract is the reference (the operation's `referenceMask`).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'standardization',
    label: 'standardization',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['zscore', 'rank']
  },
  {id: 'firstWeight', label: 'weights[0]', min: 0, max: 3, step: 0.1, default: 1},
  {id: 'secondWeight', label: 'weights[1]', min: 0, max: 3, step: 0.1, default: 1},
  {id: 'referenceTract', label: 'referenceTract', min: 0, max: 155, step: 1, default: 40}
];
