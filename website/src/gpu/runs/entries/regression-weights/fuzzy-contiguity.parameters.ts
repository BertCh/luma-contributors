// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUFuzzyContiguity's `tolerance` prop; the gap is the sliver width between the tiles.
// Measured on the 36 tracts: at an inset of 0.05 touching tracts are at most 0.0194 apart and the
// nearest pair that does not touch is 0.0231 apart, so 0.021 links neighbors and nothing else;
// past 0.0231 the slider starts to link tracts that are not neighbors.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'tolerance',
    label: 'tolerance',
    min: 0,
    max: 0.08,
    step: 0.002,
    default: 0.021
  },
  {
    id: 'gap',
    label: 'Data: boundary inset',
    min: 0,
    max: 0.25,
    step: 0.01,
    default: 0.05
  }
];
