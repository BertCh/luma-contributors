// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointPatternEnvelope's `rank` and `statistic`; clustering varies the observed data.
export const parameters: GpuSceneParameter[] = [
  {id: 'rank', label: 'rank', min: 1, max: 10, step: 1, default: 3},
  {
    id: 'statistic',
    label: 'statistic',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['maximum-deviation', 'integrated-squared-deviation', 'extreme-rank']
  },
  {id: 'clustering', label: 'Data: clustering', min: 0, max: 0.8, step: 0.05, default: 0.35}
];
