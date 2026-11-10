// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'rowCount',
    label: 'Data: vessel count',
    min: 3,
    max: 12,
    step: 1,
    default: 7,
    unit: 'vessels'
  },
  {id: 'maximumFixes', label: 'Data: fixes per vessel', min: 4, max: 72, step: 1, default: 36}
];
