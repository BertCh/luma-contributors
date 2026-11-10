// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUZonalStatistics' `sumOrder`; the output picks which statistic buffer is drawn and
// `pointCount` varies the data.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'statistic',
    label: 'output',
    min: 0,
    max: 5,
    step: 1,
    default: 1,
    options: ['counts', 'means', 'minima', 'maxima', 'standardDeviations', 'sums']
  },
  {id: 'pointCount', label: 'Data: workplace blocks', min: 100, max: 1528, step: 1, default: 1000}
];
