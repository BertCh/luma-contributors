// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainRGBDecode({encoding, alphaNoData, validRange}); the tile is re-encoded to match the encoding.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'alphaNoData',
    label: 'alphaNoData',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  },
  {
    id: 'validMaximum',
    label: 'validRange: maximum',
    min: 700,
    max: 2400,
    step: 50,
    default: 2400,
    unit: ' m'
  }
];
