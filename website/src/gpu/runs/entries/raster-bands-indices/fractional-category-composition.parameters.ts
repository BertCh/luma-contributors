// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'gridWidth', label: 'grid.width', min: 6, max: 60, step: 2, default: 14, unit: 'cells'},
  {
    id: 'threshold',
    label: 'Data: middle class threshold',
    min: 0.2,
    max: 0.8,
    step: 0.02,
    default: 0.5
  }
];
