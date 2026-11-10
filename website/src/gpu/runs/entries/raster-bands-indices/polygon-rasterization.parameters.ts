// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'gridWidth', label: 'grid.width', min: 16, max: 120, step: 4, default: 48, unit: 'cells'},
  {
    id: 'boundaryScale',
    label: 'Data: tract boundary scale',
    min: 0.3,
    max: 1,
    step: 0.05,
    default: 1
  }
];
