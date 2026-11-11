// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUContiguityWeights' `criterion` and `snapTolerance` props; the gap opens slivers.
// Measured on the 36 tracts: at an inset of 0.04 touching tracts end up at most 0.0156 apart and
// the nearest pair that does not touch is 0.0202 apart, so 0.018 links neighbors and nothing else.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'criterion',
    label: 'criterion',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['queen', 'rook']
  },
  {
    id: 'snapTolerance',
    label: 'snapTolerance',
    min: 0,
    max: 0.04,
    step: 0.002,
    default: 0.018
  },
  {
    id: 'gap',
    label: 'Data: boundary inset',
    min: 0,
    max: 0.25,
    step: 0.01,
    default: 0.04
  }
];
