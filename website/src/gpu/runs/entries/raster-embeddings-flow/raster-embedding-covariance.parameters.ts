// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'windowRadius',
    label: 'Data: window radius',
    min: 2,
    max: 14,
    step: 1,
    default: 6,
    unit: 'px'
  },
  {id: 'noise', label: 'Data: noise', min: 0, max: 0.5, step: 0.05, default: 0}
];
