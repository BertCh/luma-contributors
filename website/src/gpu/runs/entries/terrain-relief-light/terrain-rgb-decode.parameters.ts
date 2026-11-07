// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainRGBDecode({encoding, alphaNoData, validRange}); the tile is re-encoded to match the encoding.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'encoding',
    label: 'encoding',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['terrarium', 'mapbox']
  },
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
    min: 300,
    max: 1400,
    step: 50,
    default: 1400,
    unit: ' m'
  }
];
