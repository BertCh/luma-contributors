// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'districtSize',
    label: 'Data: district size',
    min: 1,
    max: 16,
    step: 1,
    default: 4,
    unit: 'tracts'
  }
];
