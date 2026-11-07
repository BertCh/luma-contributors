// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// statistic is a compile-time prop of addPointsInPolygonsChoroplethRecipe; classCount is the
// per-frame class breaks setting; the trend adds an east-west gradient to the point values.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'statistic',
    label: 'statistic',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['mean', 'sum', 'maximum']
  },
  {id: 'classCount', label: 'classCount', min: 2, max: 6, step: 1, default: 5},
  {id: 'trend', label: 'Data: eastWestTrend', min: 0, max: 2, step: 0.1, default: 0.6}
];
