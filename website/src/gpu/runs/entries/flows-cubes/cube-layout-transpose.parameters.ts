// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// values is the format of the cube that is transposed (uint32 counts or float32 means); cellCount
// is GPUCubeLayoutTranspose's compile-time size; mask changes which flashes the cube holds.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'values',
    label: 'values',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['uint32 counts', 'float32 means']
  },
  {
    id: 'cellCount',
    label: 'cellCount',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['12', '24', '40']
  },
  {id: 'mask', label: 'Data: energy at least', min: 0, max: 400, step: 10, default: 0, unit: 'fJ'}
];
