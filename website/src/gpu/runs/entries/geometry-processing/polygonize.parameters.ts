// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUPolygonize has no per-frame prop; the sliders change how many lines are drawn.
export const parameters: GpuSceneParameter[] = [
  {id: 'firstCount', label: 'Data: west-east lines', min: 1, max: 5, step: 1, default: 3},
  {id: 'secondCount', label: 'Data: south-north lines', min: 1, max: 6, step: 1, default: 4}
];
