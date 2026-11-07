// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSegregation's `groupCount`, `scales` (aspatial or queen contiguity) and
// `spatialForm`, plus how strongly the groups separate in space.
export const parameters: GpuSceneParameter[] = [
  {id: 'groupCount', label: 'groupCount', min: 2, max: 5, step: 1, default: 3},
  {
    id: 'scale',
    label: 'scales',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['aspatial', 'queen contiguity']
  },
  {
    id: 'spatialForm',
    label: 'spatialForm',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['environment', 'smoothed-population']
  },
  {id: 'separation', label: 'Data: separation', min: 0, max: 1, step: 0.05, default: 0.35}
];
