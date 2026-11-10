// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUTerrainSummitsParameterValues({radius, minimumDrop}) and GPUTerrainSummits({incompleteNeighborhood}).
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 200, max: 1200, step: 50, default: 350, unit: ' m'},
  {id: 'minimumDrop', label: 'minimumDrop', min: 0, max: 60, step: 2, default: 0, unit: ' m'},
  {id: 'noise', label: 'Data: noise', min: 0, max: 40, step: 1, default: 30, unit: ' m'},
  {
    id: 'incompleteNeighborhood',
    label: 'incompleteNeighborhood',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['reject', 'ignore']
  }
];
