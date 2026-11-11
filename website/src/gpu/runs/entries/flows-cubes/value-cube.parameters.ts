// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// statistic picks which output view (counts, sums, means) is drawn. gridSize fixes cellCount at
// compile time. mask is the per-frame row mask of GPUValueCube: flashes below the energy are skipped.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'statistic',
    label: 'output',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['counts', 'sums', 'means']
  },
  {
    id: 'gridSize',
    label: 'cellCount',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['12', '24', '40']
  },
  {id: 'mask', label: 'mask: energy at least', min: 0, max: 400, step: 10, default: 0, unit: 'fJ'}
];
