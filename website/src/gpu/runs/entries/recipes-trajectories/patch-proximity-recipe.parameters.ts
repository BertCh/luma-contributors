// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// edgeDepth and coreModel are compile-time props of addPatchProximityRecipe; the fragmentation
// slider reshapes the landscape.
export const parameters: GpuSceneParameter[] = [
  {id: 'edgeDepth', label: 'edgeDepth', min: 1, max: 5, step: 1, default: 2, unit: 'cells'},
  {
    id: 'coreModel',
    label: 'coreModel',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['physical', 'raster-erosion']
  },
  {id: 'fragmentation', label: 'Data: fragmentation', min: 2, max: 6, step: 1, default: 3}
];
