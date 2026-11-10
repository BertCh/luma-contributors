// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

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
  {id: 'grain', label: 'Data: sampling grain', min: 1, max: 6, step: 1, default: 1}
];
