// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `complement` is a GPUSphericalPolygonContains prop (compile time). `holeSize` and `size` move and resize
// the modeled survey exclusion.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'complement',
    label: 'complement',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  },
  {
    id: 'size',
    label: 'exclusion latitude',
    min: 43.0,
    max: 46.3,
    step: 0.05,
    default: 44.9,
    unit: '°'
  },
  {
    id: 'exclusionLongitude',
    label: 'exclusion longitude',
    min: -88.6,
    max: -85.4,
    step: 0.05,
    default: -87.3,
    unit: '°'
  },
  {
    id: 'holeSize',
    label: 'exclusion radius',
    min: 4,
    max: 80,
    step: 1,
    default: 65,
    unit: 'km'
  }
];
