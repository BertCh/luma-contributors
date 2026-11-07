// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// probability is the per-frame `probabilities` view of addZonalDistributionRecipe (the quantile
// drawn per zone); the other two shape the point values.
export const parameters: GpuSceneParameter[] = [
  {id: 'probability', label: 'probabilities', min: 0, max: 1, step: 0.05, default: 0.5},
  {id: 'trend', label: 'Data: eastWestTrend', min: 0, max: 2, step: 0.1, default: 1},
  {id: 'spread', label: 'Data: spread', min: 0, max: 2, step: 0.1, default: 1}
];
