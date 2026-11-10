// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'resolution', label: 'resolution', min: 11, max: 13, step: 1, default: 12},
  {
    id: 'classify',
    label: 'classify',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['delta', 'zScore', 'percentChange']
  },
  {
    id: 'quarter',
    label: 'quarters',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['Jan-Mar to Apr-Jun', 'Apr-Jun to Jul-Sep', 'Jul-Sep to Oct-Dec']
  }
];
