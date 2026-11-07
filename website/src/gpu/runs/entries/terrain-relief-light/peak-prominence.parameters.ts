// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUTerrainPeakProminence has no props besides its views; the sliders change the input surface.
export const parameters: GpuSceneParameter[] = [
  {id: 'relief', label: 'Data: relief', min: 0.4, max: 2, step: 0.1, default: 1},
  {id: 'noise', label: 'Data: noise', min: 0, max: 20, step: 1, default: 20, unit: ' m'},
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['prominence', 'keySaddle']
  }
];
