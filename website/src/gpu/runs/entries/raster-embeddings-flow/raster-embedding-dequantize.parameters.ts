// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'band',
    label: 'Band shown',
    min: 0,
    max: 5,
    step: 1,
    default: 3,
    options: ['blue', 'green', 'red', 'NIR', 'SWIR1', 'SWIR2']
  },
  {id: 'noData', label: 'Data: NoData pixels', min: 0, max: 0.5, step: 0.05, default: 0.1}
];
