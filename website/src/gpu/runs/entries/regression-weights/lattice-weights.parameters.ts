// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPULatticeWeights' `criterion`, `radius` and `mask` props.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'criterion',
    label: 'criterion',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['rook', 'queen']
  },
  {id: 'radius', label: 'radius', min: 1, max: 2, step: 1, default: 1},
  {
    id: 'minimumElevation',
    label: 'Mask: minimum elevation',
    min: 700,
    max: 2000,
    step: 50,
    default: 1000,
    unit: ' m'
  }
];
