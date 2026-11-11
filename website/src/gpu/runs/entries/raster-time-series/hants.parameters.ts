// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'fitErrorTolerance',
    label: 'fitErrorTolerance',
    min: 0.02,
    max: 0.2,
    step: 0.01,
    default: 0.07
  },
  {id: 'harmonicCount', label: 'harmonicCount', min: 1, max: 3, step: 1, default: 2},
  {id: 'rejectionPassCount', label: 'rejectionPassCount', min: 2, max: 16, step: 2, default: 16},
  {id: 'cloudFraction', label: 'Data: cloud fraction', min: 0, max: 0.4, step: 0.02, default: 0.12}
];
