// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// radius and temporalWindow are per-frame settings of addSpaceTimeHotSpotsRecipe; the growth
// slider changes the events (and so rebuilds).
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 1, max: 3, step: 0.5, default: 1.5, unit: 'cells'},
  {
    id: 'temporalWindow',
    label: 'temporalWindow',
    min: 1,
    max: 3,
    step: 1,
    default: 1,
    unit: 'slices'
  },
  {id: 'growth', label: 'Data: monthlyGrowth', min: 0, max: 4, step: 1, default: 1}
];
