// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'criticalValue', label: 'criticalValue', min: 1, max: 8, step: 0.25, default: 4},
  {id: 'bandwidth', label: 'bandwidth', min: 0.2, max: 1, step: 0.05, default: 0.5},
  {
    id: 'historySliceCount',
    label: 'historySliceCount',
    min: 5,
    max: 8,
    step: 1,
    default: 7
  }
];
