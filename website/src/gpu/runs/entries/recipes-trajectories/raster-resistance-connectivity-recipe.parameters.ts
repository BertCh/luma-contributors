// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// source and neighborhood are compile-time props of addRasterResistanceConnectivityRecipe (the
// pointer settles `source` on one of five habitat patches); the barrier resistance rewrites the
// resistance raster, so it never recompiles.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'source',
    label: 'source',
    min: 0,
    max: 4,
    step: 1,
    default: 2,
    options: ['patch 1', 'patch 2', 'patch 3', 'patch 4', 'patch 5']
  },
  {
    id: 'neighborhood',
    label: 'neighborhood',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['4', '8']
  },
  {
    id: 'barrierResistance',
    label: 'Data: barrier resistance',
    min: 10,
    max: 200,
    step: 10,
    default: 60
  }
];
