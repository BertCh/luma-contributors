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
    default: 1,
    options: ['unconstrained', 'origin', 'destination', 'doubly']
  },
  {id: 'decayScale', label: 'Coefficient search scale', min: 0.2, max: 3, step: 0.2, default: 1}
];
