// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'patchCount', label: 'Data: patches', min: 6, max: 40, step: 1, default: 18},
  {
    id: 'countRasterBorder',
    label: 'countRasterBorder',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  }
];
