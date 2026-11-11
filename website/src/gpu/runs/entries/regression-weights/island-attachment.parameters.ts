// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUIslandAttachment's `reciprocal` prop; the band radius decides which points are islands.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'reciprocal',
    label: 'reciprocal',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  },
  {
    id: 'radius',
    label: 'Data: band radius',
    min: 0.02,
    max: 0.12,
    step: 0.005,
    default: 0.04
  }
];
