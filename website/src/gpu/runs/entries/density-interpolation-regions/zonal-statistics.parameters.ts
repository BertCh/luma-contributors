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
  {
    id: 'sumOrder',
    label: 'sumOrder',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['atomic', 'sorted']
  },
  {id: 'pointCount', label: 'Data: pointCount', min: 500, max: 6000, step: 250, default: 3000}
];
