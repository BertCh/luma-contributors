// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// orientation is a constructor prop of GPUGeometryValidity; defects damages that many polygons.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'orientation',
    label: 'orientation',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['counter-clockwise-shell', 'clockwise-shell', 'ignore']
  },
  {id: 'defects', label: 'Data: defects', min: 0, max: 12, step: 1, default: 6}
];

export const legend: RunLegendItem[] = [
  {label: 'valid', color: '#7388b8'},
  {label: 'self-intersection', color: '#ff6b6b'},
  {label: 'hole outside or rings crossing', color: '#f5d547'},
  {label: 'wrong orientation only', color: '#4fd1c5'},
  {label: 'other (repeated vertex)', color: '#c792ea'}
];
