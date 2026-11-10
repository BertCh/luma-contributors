// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// target is the one-row per-frame view that names the cell to trace back from, here as a position on
// the map; the contrast is the friction raster's roughness (the surface the route follows).
export const parameters: GpuSceneParameter[] = [
  {id: 'targetColumn', label: 'target (x)', min: 0, max: 99, step: 1, default: 90, unit: '%'},
  {id: 'targetRow', label: 'target (y)', min: 0, max: 99, step: 1, default: 24, unit: '%'},
  {id: 'contrast', label: 'Data: friction contrast', min: 0, max: 30, step: 1, default: 7}
];
