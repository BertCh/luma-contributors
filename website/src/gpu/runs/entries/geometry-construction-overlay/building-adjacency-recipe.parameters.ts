// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'splitEvery',
    label: 'Data: splitEvery',
    min: 2,
    max: 8,
    step: 1,
    default: 4,
    unit: 'buildings'
  },
  {id: 'gap', label: 'Data: gap', min: 0, max: 0.04, step: 0.002, default: 0.02}
];
