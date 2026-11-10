// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUFuzzyContiguity's `tolerance` prop; the gap is the sliver width between the tiles.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'tolerance',
    label: 'tolerance',
    min: 0,
    max: 0.08,
    step: 0.002,
    default: 0.03
  },
  {
    id: 'gap',
    label: 'Data: boundary inset',
    min: 0,
    max: 0.25,
    step: 0.01,
    default: 0.1
  }
];
