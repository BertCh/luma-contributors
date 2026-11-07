// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `complement` is a GPUSphericalPolygonContains prop (compile time). `holeSize` and `size` rewrite
// the polygon rings per frame.
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
  {id: 'size', label: 'Data: size', min: 0.25, max: 0.5, step: 0.01, default: 0.42},
  {id: 'holeSize', label: 'Data: holeSize', min: 0, max: 0.25, step: 0.01, default: 0.14}
];
