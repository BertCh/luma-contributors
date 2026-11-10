// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUSolarIrradianceSunTable({latitude, start}) (a day in the chosen month) and GPUSolarIrradiance({output}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'latitude',
    label: 'sunTable: latitude',
    min: -60,
    max: 70,
    step: 1,
    default: 36,
    unit: '°'
  },
  {id: 'month', label: 'sunTable: month', min: 1, max: 12, step: 1, default: 12},
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['sunHours', 'insolation']
  }
];
