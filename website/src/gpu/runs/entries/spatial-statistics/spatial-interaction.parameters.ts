// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialInteraction's `constraint`, plus the flows it is fitted to.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'constraint',
    label: 'constraint',
    min: 0,
    max: 3,
    step: 1,
    default: 3,
    options: ['unconstrained', 'origin', 'destination', 'doubly']
  },
  {id: 'distanceDecay', label: 'Data: distanceDecay', min: 0.5, max: 3, step: 0.1, default: 1.5},
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.3}
];
