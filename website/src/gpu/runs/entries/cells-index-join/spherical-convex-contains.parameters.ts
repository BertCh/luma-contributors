// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `complement` is a GPUSphericalConvexContains prop (compile time). `sides` and `size` shape the
// geodesic search area; `size` is a per-frame rewrite of the polygon vertices.
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
  {id: 'sides', label: 'sides', min: 3, max: 8, step: 1, default: 6},
  {id: 'size', label: 'search radius', min: 15, max: 120, step: 1, default: 60, unit: 'km'}
];
