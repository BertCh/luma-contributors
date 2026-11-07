// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `operation` is a GPUSphericalPolygonOverlay prop (compile time). `shift` slides the right
// polygon over the left one (a per-frame rewrite of its vertices).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['intersection', 'union', 'difference', 'symmetric-difference']
  },
  {id: 'shift', label: 'Data: shift', min: -0.4, max: 0.4, step: 0.01, default: 0.18}
];
