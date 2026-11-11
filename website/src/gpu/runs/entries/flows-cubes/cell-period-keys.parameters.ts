// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// resolution is the H3 resolution of the cell keys and periodCount the size of the period axis; both
// are compile-time. mask is the per-frame row mask: flashes below the energy are not packed.
export const parameters: GpuSceneParameter[] = [
  {id: 'resolution', label: 'H3 resolution', min: 1, max: 15, step: 1, default: 3},
  {
    id: 'periodCount',
    label: 'periodCount',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['6', '18', '72']
  },
  {id: 'mask', label: 'mask: energy at least', min: 0, max: 400, step: 10, default: 0, unit: 'fJ'}
];
