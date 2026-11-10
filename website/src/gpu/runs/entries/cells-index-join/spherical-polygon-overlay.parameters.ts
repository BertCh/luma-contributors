// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `operation` is a GPUSphericalPolygonOverlay prop (compile time). `shift` and `shiftLatitude` slide
// the right polygon over the left one (a per-frame rewrite of its vertices).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 3,
    step: 1,
    default: 2,
    options: ['intersection', 'union', 'difference', 'symmetric-difference']
  },
  {
    id: 'shift',
    label: 'corridor longitude',
    min: -90.5,
    max: -85.8,
    step: 0.05,
    default: -87.8,
    unit: '°'
  },
  {
    id: 'shiftLatitude',
    label: 'corridor latitude',
    min: 46.8,
    max: 48.6,
    step: 0.05,
    default: 47.7,
    unit: '°'
  }
];
