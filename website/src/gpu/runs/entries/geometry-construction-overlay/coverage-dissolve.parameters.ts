// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'districtSize',
    label: 'Data: districtSize',
    min: 1,
    max: 4,
    step: 1,
    default: 2,
    unit: 'cells'
  }
];
