// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `complement` is a GPUSphericalConvexContains prop (compile time). `sides` and `size` shape the two
// convex spherical polygons; `size` is a per-frame rewrite of the polygon vertices.
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
  {id: 'sides', label: 'Data: sides', min: 3, max: 8, step: 1, default: 6},
  {id: 'size', label: 'Data: size', min: 0.2, max: 0.48, step: 0.01, default: 0.36}
];
