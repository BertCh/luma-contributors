// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUCellCover's compile-time `family` and `resolution`; `size` scales the input polygons (a
// per-frame rewrite of `positions`).
// Core cells (provably inside) are drawn in one color and border cells in another.
export const parameters: GpuSceneParameter[] = [
  {id: 'family', label: 'family', min: 0, max: 1, step: 1, default: 0, options: ['h3', 'quadbin']},
  {id: 'resolution', label: 'resolution', min: 4, max: 7, step: 1, default: 6},
  {id: 'size', label: 'Data: size', min: 0.5, max: 1.5, step: 0.05, default: 1}
];
