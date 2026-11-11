// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'direction',
    label: 'Datum transformation',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['NAD27 → NAD83', 'NAD83 → NAD27']
  },
  {id: 'maximumIterations', label: 'Inverse iteration limit', min: 1, max: 4, step: 1, default: 3},
  {
    id: 'magnification',
    label: 'Vector magnification',
    unit: '×',
    min: 1,
    max: 3000,
    step: 1,
    default: 1500
  }
];

export const legend: RunLegendItem[] = [
  {label: '0 m shift', color: 'light-dark(#f6f4ef, #0b0f1a)'},
  {label: '30 m or more', color: 'light-dark(#3b0f5e, #fff1a8)'},
  {label: 'Magnified shift vector', color: 'var(--color-text)'}
];
