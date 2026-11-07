// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointCloudStatisticalFilter's `standardDeviations` and the neighborhood `capacity`
// it reads, plus the number of stray points added to the cloud.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'standardDeviations',
    label: 'standardDeviations',
    min: 0.5,
    max: 4,
    step: 0.25,
    default: 1.5
  },
  {id: 'capacity', label: 'capacity', min: 4, max: 24, step: 2, default: 12},
  {id: 'strays', label: 'Data: stray points', min: 0, max: 300, step: 20, default: 160}
];
